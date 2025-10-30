"""
eICU Collaborative Research Database Data Loading Module

This module handles loading and initial filtering of eICU data.
Supports gzipped CSV files from the eICU Collaborative Research Database.

eICU is a publicly available critical care database containing
de-identified health data from multiple ICUs. It includes:
- patient: Demographics and ICU stay information
- lab: Laboratory test results
- diagnosis: Diagnosis information
- medication: Medication administration records

Reference: https://eicu-crd.mit.edu/
"""

import logging
import pandas as pd
import gzip
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

# ============================================================================
# Core eICU Table Loaders
# ============================================================================

class eICULoader:
    """
    Interface for loading eICU Collaborative Research Database data.

    Rationale:
        eICU is distributed as gzipped CSV files. This class provides
        a consistent interface to load and normalize the data similar
        to the MIMIC-III loader.
    """

    def __init__(
        self,
        data_dir: Union[str, Path]
    ):
        """
        Initialize eICU data loader.

        Args:
            data_dir: Path to directory containing eICU CSV files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.logger.info(f"Initialized eICU loader with data_dir: {self.data_dir}")

    def _load_csv(self, table_name: str) -> pd.DataFrame:
        """
        Load a gzipped CSV file from eICU dataset.

        Args:
            table_name: Name of the table (e.g., 'patient', 'lab')

        Returns:
            DataFrame with the table contents
        """
        csv_path = self.data_dir / f"{table_name}.csv.gz"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.logger.info(f"Loading {table_name} from {csv_path}")

        # Read gzipped CSV
        with gzip.open(csv_path, 'rt') as f:
            df = pd.read_csv(f, low_memory=False)

        self.logger.info(f"Loaded {len(df):,} rows from {table_name}")
        return df

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load any eICU table by name.

        Args:
            table_name: Name of the table

        Returns:
            DataFrame with the table contents
        """
        return self._load_csv(table_name)

    def load_patients(self) -> pd.DataFrame:
        """
        Load patient demographics and ICU stay information.

        Returns:
            DataFrame with columns:
                - patientunitstayid: Unique ICU stay identifier
                - uniquepid: Unique patient identifier (across multiple stays)
                - gender: Patient gender
                - age: Patient age (may be '>89' for privacy)
                - ethnicity: Patient ethnicity
                - hospitalid: Hospital identifier
                - unittype: Type of ICU unit
        """
        df = self._load_csv('patient')
        self.logger.info(f"Loaded {len(df):,} patient stays")
        return df

    def load_lab(self, chunksize: Optional[int] = None) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load laboratory test results.

        Args:
            chunksize: If specified, returns iterator that yields DataFrames
                      of this size (useful for large datasets)

        Returns:
            DataFrame (or iterator) with columns:
                - patientunitstayid: Patient stay identifier
                - labresultoffset: Minutes from unit admission
                - labname: Name of the lab test
                - labresult: Numeric result value
                - labresulttext: Text representation of result
        """
        csv_path = self.data_dir / "lab.csv.gz"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if chunksize is not None:
            self.logger.info(f"Loading lab in chunks of {chunksize:,} rows")
            with gzip.open(csv_path, 'rt') as f:
                return pd.read_csv(f, chunksize=chunksize, low_memory=False)
        else:
            df = self._load_csv('lab')
            self.logger.info(f"Loaded {len(df):,} lab results")
            return df

    def load_diagnosis(self) -> pd.DataFrame:
        """
        Load diagnosis information.

        Returns:
            DataFrame with columns:
                - patientunitstayid: Patient stay identifier
                - diagnosisoffset: Minutes from unit admission
                - diagnosisstring: Hierarchical diagnosis string
                - icd9code: ICD-9 diagnosis code
                - diagnosispriority: Priority level of diagnosis
        """
        df = self._load_csv('diagnosis')
        self.logger.info(f"Loaded {len(df):,} diagnoses")
        return df

    def load_medication(self) -> pd.DataFrame:
        """
        Load medication administration records.

        Returns:
            DataFrame with columns:
                - patientunitstayid: Patient stay identifier
                - drugstartoffset: Minutes from unit admission when drug started
                - drugname: Name of the medication
                - dosage: Dosage information
                - routeadmin: Route of administration
                - frequency: Frequency of administration
        """
        df = self._load_csv('medication')
        self.logger.info(f"Loaded {len(df):,} medication records")
        return df

    # Compatibility methods that return MIMIC-format data
    def load_labevents(self, chunksize: Optional[int] = None):
        """
        Load lab events in MIMIC-III compatible format.

        Returns lab data with columns: SUBJECT_ID, ITEMID, VALUENUM, CHARTTIME
        """
        labs = self.load_lab(chunksize=chunksize)

        # Map to MIMIC format
        labs_mapped = labs.copy()
        labs_mapped['SUBJECT_ID'] = labs_mapped['patientunitstayid']
        labs_mapped['ITEMID'] = labs_mapped['labname']  # Use lab name as item ID
        labs_mapped['VALUENUM'] = pd.to_numeric(labs_mapped['labresult'], errors='coerce')
        labs_mapped['CHARTTIME'] = labs_mapped['labresultoffset']  # Keep as offset in minutes

        return labs_mapped

    def load_diagnoses_icd(self) -> pd.DataFrame:
        """
        Load diagnoses in MIMIC-III compatible format.

        Returns diagnosis data with columns: SUBJECT_ID, HADM_ID, ICD9_CODE
        """
        diagnoses = self.load_diagnosis()

        # Load patient table to get patienthealthsystemstayid mapping
        patients = self.load_patients()

        # Create mapping from patientunitstayid to patienthealthsystemstayid
        id_mapping = patients[['patientunitstayid', 'patienthealthsystemstayid']].drop_duplicates()

        # Map to MIMIC format
        diagnoses_mapped = diagnoses.copy()
        diagnoses_mapped['SUBJECT_ID'] = diagnoses_mapped['patientunitstayid']

        # Join to get HADM_ID (patienthealthsystemstayid)
        diagnoses_mapped = diagnoses_mapped.merge(
            id_mapping,
            on='patientunitstayid',
            how='left'
        )
        diagnoses_mapped['HADM_ID'] = diagnoses_mapped['patienthealthsystemstayid']

        # eICU has comma-separated ICD9 codes, take the first one
        def extract_first_icd9(icd9_str):
            if pd.isna(icd9_str):
                return None
            # Split by comma and take first code
            codes = str(icd9_str).split(',')
            return codes[0].strip() if codes else None

        diagnoses_mapped['ICD9_CODE'] = diagnoses_mapped['icd9code'].apply(extract_first_icd9)

        # If no ICD9 code, use diagnosis string
        diagnoses_mapped['ICD9_CODE'] = diagnoses_mapped['ICD9_CODE'].fillna(
            diagnoses_mapped['diagnosisstring']
        )

        # Parse diagnosis hierarchy from diagnosisstring
        # Format: "category|subcategory|specific"
        def parse_diagnosis_category(diagnosisstring):
            if pd.isna(diagnosisstring):
                return 'Unknown'
            parts = str(diagnosisstring).split('|')
            return parts[0].strip() if parts else 'Unknown'

        def parse_diagnosis_subcategory(diagnosisstring):
            if pd.isna(diagnosisstring):
                return 'Unknown'
            parts = str(diagnosisstring).split('|')
            return parts[1].strip() if len(parts) > 1 else 'Unknown'

        diagnoses_mapped['DIAGNOSIS_CATEGORY'] = diagnoses_mapped['diagnosisstring'].apply(
            parse_diagnosis_category
        )
        diagnoses_mapped['DIAGNOSIS_SUBCATEGORY'] = diagnoses_mapped['diagnosisstring'].apply(
            parse_diagnosis_subcategory
        )

        # Add diagnosis priority (Primary, Major, Other)
        diagnoses_mapped['DIAGNOSIS_PRIORITY'] = diagnoses_mapped['diagnosispriority'].fillna('Other')

        return diagnoses_mapped

    def load_prescriptions(self) -> pd.DataFrame:
        """
        Load medications in MIMIC-III compatible format.

        Returns prescription data with columns: SUBJECT_ID, HADM_ID, DRUG
        """
        medications = self.load_medication()

        # Load patient table to get patienthealthsystemstayid mapping
        patients = self.load_patients()

        # Create mapping from patientunitstayid to patienthealthsystemstayid
        id_mapping = patients[['patientunitstayid', 'patienthealthsystemstayid']].drop_duplicates()

        # Map to MIMIC format
        medications_mapped = medications.copy()
        medications_mapped['SUBJECT_ID'] = medications_mapped['patientunitstayid']

        # Join to get HADM_ID (patienthealthsystemstayid)
        medications_mapped = medications_mapped.merge(
            id_mapping,
            on='patientunitstayid',
            how='left'
        )
        medications_mapped['HADM_ID'] = medications_mapped['patienthealthsystemstayid']
        medications_mapped['DRUG'] = medications_mapped['drugname']

        # Extract medication metadata features
        # Route of administration
        medications_mapped['ROUTE'] = medications_mapped['routeadmin'].fillna('Unknown')

        # Frequency (dosing schedule)
        medications_mapped['FREQUENCY'] = medications_mapped['frequency'].fillna('Unknown')

        # PRN (as needed) flag
        medications_mapped['PRN'] = medications_mapped['prn'].fillna('No')

        # IV admixture flag
        medications_mapped['IV_ADMIXTURE'] = medications_mapped['drugivadmixture'].fillna('No')

        # Dosage
        medications_mapped['DOSAGE'] = medications_mapped['dosage'].fillna('')

        return medications_mapped

    def load_apache(self) -> pd.DataFrame:
        """
        Load APACHE severity scores from apachePatientResult table.

        Returns:
            DataFrame with columns:
                - patientunitstayid: ICU stay identifier
                - acutephysiologyscore: Acute physiology score (0-299)
                - apachescore: Overall APACHE score (0-299)
                - predictedicumortality: Predicted ICU mortality (0-1)
                - predictedhospitalmortality: Predicted hospital mortality (0-1)
                - physicianspeciality: Physician specialty
        """
        apache = self._load_csv('apachePatientResult')
        self.logger.info(f"Loaded {len(apache):,} APACHE score records")
        return apache

    def load_apache_for_cohort(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """
        Load APACHE scores and map to MIMIC-III compatible format.

        Args:
            cohort: Patient cohort DataFrame with SUBJECT_ID and HADM_ID

        Returns:
            DataFrame with APACHE scores mapped to SUBJECT_ID
        """
        apache = self.load_apache()

        # Map to cohort using patientunitstayid
        # Note: eICU uses 'physicianspeciality' (British spelling)
        apache_mapped = apache[[
            'patientunitstayid',
            'acutephysiologyscore',
            'apachescore',
            'predictedicumortality',
            'predictedhospitalmortality'
        ]].copy()

        # Rename for consistency
        apache_mapped['SUBJECT_ID'] = apache_mapped['patientunitstayid']

        return apache_mapped

    def load_d_labitems(self) -> pd.DataFrame:
        """
        Create a lab items dictionary (similar to MIMIC's D_LABITEMS).

        In eICU, lab names are embedded in the lab table, so we extract
        unique lab names and create a compatible format.
        """
        labs = self.load_lab()
        lab_names = labs['labname'].dropna().unique()

        labitems = pd.DataFrame({
            'ITEMID': lab_names,
            'LABEL': lab_names,
            'FLUID': 'Blood',  # Default assumption
            'CATEGORY': 'Chemistry'  # Default assumption
        })

        self.logger.info(f"Created labitems dictionary with {len(labitems)} unique labs")
        return labitems


# ============================================================================
# Data Validation and Quality Checks
# ============================================================================

def validate_eicu_data(loader: eICULoader) -> Dict[str, any]:
    """
    Validate loaded eICU data for completeness and quality.

    Args:
        loader: Initialized eICU loader

    Returns:
        Dictionary with validation statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating eICU data...")

    stats = {}

    # Load and validate patients
    patients = loader.load_patients()
    stats['n_patient_stays'] = len(patients)
    stats['n_unique_patients'] = patients['uniquepid'].nunique()
    stats['missing_gender'] = patients['gender'].isna().sum()
    stats['missing_age'] = patients['age'].isna().sum()

    # Load and validate labs
    labs = loader.load_lab()
    stats['n_lab_results'] = len(labs)
    stats['n_unique_lab_types'] = labs['labname'].nunique()
    stats['missing_lab_values'] = labs['labresult'].isna().sum()

    # Load and validate diagnoses
    diagnoses = loader.load_diagnosis()
    stats['n_diagnoses'] = len(diagnoses)
    stats['n_unique_diagnosis_strings'] = diagnoses['diagnosisstring'].nunique()

    # Load and validate medications
    medications = loader.load_medication()
    stats['n_medications'] = len(medications)
    stats['n_unique_drugs'] = medications['drugname'].nunique()

    logger.info("Validation complete:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:,}")

    return stats


# ============================================================================
# Mapping Functions to MIMIC-III Format
# ============================================================================

def map_eicu_to_mimic_format(loader: eICULoader) -> Dict[str, pd.DataFrame]:
    """
    Map eICU data to MIMIC-III-like format for compatibility with existing pipeline.

    This function transforms eICU tables to match the structure expected by
    the preprocessing and graph building steps.

    Args:
        loader: Initialized eICU loader

    Returns:
        Dictionary with keys:
            - 'patients': Patient demographics (mapped to MIMIC format)
            - 'admissions': ICU stay information
            - 'labevents': Laboratory results
            - 'diagnoses': Diagnoses
            - 'prescriptions': Medications
            - 'labitems': Lab test dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Mapping eICU data to MIMIC-III format...")

    # Load raw eICU data
    patients = loader.load_patients()
    labs = loader.load_lab()
    diagnoses = loader.load_diagnosis()
    medications = loader.load_medication()

    # Map patients table
    # eICU uses 'patientunitstayid' while MIMIC uses 'SUBJECT_ID'
    patients_mapped = patients.copy()
    patients_mapped['SUBJECT_ID'] = patients_mapped['patientunitstayid']
    patients_mapped['GENDER'] = patients_mapped['gender']

    # Handle age - convert '>89' to 90, and convert string ages to numeric
    def parse_age(age):
        if pd.isna(age):
            return None
        age_str = str(age).strip()
        if age_str == '> 89':
            return 90
        try:
            return int(age_str)
        except:
            return None

    patients_mapped['AGE'] = patients_mapped['age'].apply(parse_age)

    # Create admissions-like table with ICU stay info
    admissions_mapped = patients.copy()
    admissions_mapped['SUBJECT_ID'] = admissions_mapped['patientunitstayid']
    admissions_mapped['HADM_ID'] = admissions_mapped['patienthealthsystemstayid']

    # Map lab events
    labs_mapped = labs.copy()
    labs_mapped['SUBJECT_ID'] = labs_mapped['patientunitstayid']
    labs_mapped['ITEMID'] = labs_mapped['labname']  # Use lab name as item ID
    labs_mapped['VALUENUM'] = pd.to_numeric(labs_mapped['labresult'], errors='coerce')
    labs_mapped['CHARTTIME'] = labs_mapped['labresultoffset']  # Keep as offset

    # Create labitems dictionary
    lab_names = labs['labname'].dropna().unique()
    labitems = pd.DataFrame({
        'ITEMID': lab_names,
        'LABEL': lab_names,
        'FLUID': 'Blood',  # Default assumption
        'CATEGORY': 'Chemistry'  # Default assumption
    })

    # Map diagnoses
    diagnoses_mapped = diagnoses.copy()
    diagnoses_mapped['SUBJECT_ID'] = diagnoses_mapped['patientunitstayid']
    diagnoses_mapped['ICD9_CODE'] = diagnoses_mapped['icd9code'].fillna(diagnoses_mapped['diagnosisstring'])

    # Map medications (prescriptions)
    medications_mapped = medications.copy()
    medications_mapped['SUBJECT_ID'] = medications_mapped['patientunitstayid']
    medications_mapped['DRUG'] = medications_mapped['drugname']
    medications_mapped['STARTDATE'] = medications_mapped['drugstartoffset']

    logger.info("Mapping complete!")

    return {
        'patients': patients_mapped,
        'admissions': admissions_mapped,
        'labevents': labs_mapped,
        'labitems': labitems,
        'diagnoses': diagnoses_mapped,
        'prescriptions': medications_mapped
    }


# ============================================================================
# Cohort Selection
# ============================================================================

def select_cohort(
    patients: pd.DataFrame,
    age_min: int = 18,
    age_max: Optional[int] = None,
    use_first_icu_only: bool = True,
    subject_limit: Optional[int] = None,
    min_los_hours: Optional[float] = None,
    exclude_deaths: bool = False,
    **kwargs  # Catch any extra config params
) -> pd.DataFrame:
    """
    Select cohort of patients from eICU dataset based on inclusion criteria.

    Args:
        patients: Patient DataFrame from eICU
        age_min: Minimum age for inclusion
        age_max: Maximum age for inclusion (None = no limit)
        use_first_icu_only: If True, only use first ICU stay per patient
        subject_limit: Maximum number of patients (None = no limit)
        min_los_hours: Minimum length of stay in hours (None = no limit)
        exclude_deaths: If True, exclude patients who died

    Returns:
        DataFrame with selected cohort
    """
    logger = logging.getLogger(__name__)
    logger.info("Selecting cohort from eICU data...")
    logger.info(f"Initial patient stays: {len(patients):,}")

    cohort = patients.copy()

    # Parse age (handle '>89' case)
    def parse_age(age):
        if pd.isna(age):
            return None
        age_str = str(age).strip()
        if age_str == '> 89':
            return 90
        try:
            return int(age_str)
        except:
            return None

    cohort['AGE'] = cohort['age'].apply(parse_age)

    # Filter by age
    age_mask = cohort['AGE'] >= age_min
    if age_max is not None:
        age_mask &= cohort['AGE'] <= age_max

    cohort = cohort[age_mask]
    logger.info(f"After age filter ({age_min}-{age_max}): {len(cohort):,}")

    # Calculate length of stay in hours
    # In eICU, unitdischargeoffset is minutes from unit admission (which is time 0)
    cohort['LOS_HOURS'] = cohort['unitdischargeoffset'] / 60.0

    # Filter by minimum length of stay
    if min_los_hours is not None:
        cohort = cohort[cohort['LOS_HOURS'] >= min_los_hours]
        logger.info(f"After LOS filter (>={min_los_hours}h): {len(cohort):,}")

    # Exclude deaths
    if exclude_deaths:
        cohort = cohort[cohort['unitdischargestatus'] == 'Alive']
        logger.info(f"After excluding deaths: {len(cohort):,}")

    # Use only first ICU stay per patient
    if use_first_icu_only:
        # Sort by unique patient ID and unit admit time
        cohort['unitadmittime24_dt'] = pd.to_datetime(
            cohort['unitadmittime24'],
            format='%H:%M:%S',
            errors='coerce'
        )
        cohort = cohort.sort_values(['uniquepid', 'unitadmittime24_dt'])
        cohort = cohort.groupby('uniquepid').first().reset_index()
        logger.info(f"After first ICU stay only: {len(cohort):,}")

    # Limit number of subjects
    if subject_limit is not None and subject_limit < len(cohort):
        cohort = cohort.head(subject_limit)
        logger.info(f"After subject limit ({subject_limit}): {len(cohort):,}")

    # Add compatibility columns for pipeline
    cohort['SUBJECT_ID'] = cohort['patientunitstayid']
    cohort['HADM_ID'] = cohort['patienthealthsystemstayid']  # Hospital stay ID
    cohort['GENDER'] = cohort['gender']

    logger.info(f"Final cohort size: {len(cohort):,}")

    return cohort
