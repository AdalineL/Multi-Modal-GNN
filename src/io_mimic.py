"""
MIMIC-III Data Loading Module

This module handles loading and initial filtering of MIMIC-III data.
Supports both CSV files and PostgreSQL database connections.

MIMIC-III is a publicly available critical care database containing
de-identified health data for ICU patients. It includes:
- PATIENTS: Demographics and mortality
- ADMISSIONS: Hospital admission/discharge info
- ICUSTAYS: ICU stay details
- LABEVENTS: Laboratory test results
- DIAGNOSES_ICD: Diagnosis codes
- PRESCRIPTIONS: Medication orders
- D_LABITEMS: Lab test dictionary
- D_ICD_DIAGNOSES: Diagnosis code dictionary

Reference: https://mimic.mit.edu/
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from datetime import datetime


# ============================================================================
# Core MIMIC-III Table Loaders
# ============================================================================

class MIMICLoader:
    """
    Unified interface for loading MIMIC-III data from CSV or database.

    Rationale:
        MIMIC-III is distributed in two formats:
        1. CSV files (easier to work with, used in demo)
        2. PostgreSQL database (for full dataset, more efficient)

        This class abstracts the source, providing consistent DataFrames.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        source: str = "csv",
        db_connection: Optional[str] = None
    ):
        """
        Args:
            data_dir: Directory containing MIMIC-III CSV files
            source: "csv" or "postgres"
            db_connection: SQLAlchemy connection string (if using postgres)
        """
        self.data_dir = Path(data_dir)
        self.source = source
        self.db_connection = db_connection

        if source == "csv" and not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if source == "postgres" and db_connection is None:
            raise ValueError("Must provide db_connection for postgres source")

        logging.info(f"MIMIC-III loader initialized (source: {source})")

    def _load_csv(self, table_name: str) -> pd.DataFrame:
        """
        Load a table from CSV file.

        Args:
            table_name: Name of MIMIC-III table (e.g., "PATIENTS")

        Returns:
            DataFrame
        """
        # MIMIC-III CSV files are usually uppercase
        csv_path = self.data_dir / f"{table_name}.csv"

        # Try lowercase if uppercase not found
        if not csv_path.exists():
            csv_path = self.data_dir / f"{table_name.lower()}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {table_name}")

        # Use low_memory=False to avoid dtype warnings
        df = pd.read_csv(csv_path, low_memory=False)

        # Normalize column names to uppercase for consistency
        df.columns = df.columns.str.upper()

        logging.info(f"Loaded {table_name} from CSV: {len(df)} rows")
        return df

    def _load_postgres(self, table_name: str) -> pd.DataFrame:
        """
        Load a table from PostgreSQL database.

        Args:
            table_name: Name of MIMIC-III table

        Returns:
            DataFrame
        """
        import sqlalchemy

        engine = sqlalchemy.create_engine(self.db_connection)

        # MIMIC-III schema in postgres is usually 'mimiciii'
        query = f"SELECT * FROM mimiciii.{table_name.lower()}"

        df = pd.read_sql(query, engine)

        logging.info(f"Loaded {table_name} from PostgreSQL: {len(df)} rows")
        return df

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load a MIMIC-III table (source-agnostic).

        Args:
            table_name: Name of table (e.g., "PATIENTS", "LABEVENTS")

        Returns:
            DataFrame
        """
        if self.source == "csv":
            return self._load_csv(table_name)
        elif self.source == "postgres":
            return self._load_postgres(table_name)
        else:
            raise ValueError(f"Unknown source: {self.source}")

    # ========================================================================
    # Convenience methods for specific tables
    # ========================================================================

    def load_patients(self) -> pd.DataFrame:
        """
        Load PATIENTS table.

        Contains:
        - SUBJECT_ID: Unique patient identifier
        - GENDER: M/F
        - DOB: Date of birth
        - DOD: Date of death (if applicable)
        """
        df = self.load_table("PATIENTS")

        # Convert dates
        date_cols = ['DOB', 'DOD']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def load_admissions(self) -> pd.DataFrame:
        """
        Load ADMISSIONS table.

        Contains:
        - SUBJECT_ID: Patient ID
        - HADM_ID: Hospital admission ID
        - ADMITTIME: Admission timestamp
        - DISCHTIME: Discharge timestamp
        - ADMISSION_TYPE: EMERGENCY, ELECTIVE, etc.
        - ETHNICITY: Patient ethnicity
        - HOSPITAL_EXPIRE_FLAG: Died in hospital (0/1)
        """
        df = self.load_table("ADMISSIONS")

        # Convert timestamps
        time_cols = ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME']
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def load_icustays(self) -> pd.DataFrame:
        """
        Load ICUSTAYS table.

        Contains:
        - SUBJECT_ID: Patient ID
        - HADM_ID: Hospital admission ID
        - ICUSTAY_ID: ICU stay ID
        - INTIME: ICU admission time
        - OUTTIME: ICU discharge time
        - LOS: Length of stay (days)
        """
        df = self.load_table("ICUSTAYS")

        # Convert timestamps
        time_cols = ['INTIME', 'OUTTIME']
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def load_labevents(self, chunksize: Optional[int] = None) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """
        Load LABEVENTS table.

        WARNING: This is the largest table in MIMIC-III (~27 million rows).
        For full dataset, consider using chunksize parameter.

        Contains:
        - SUBJECT_ID: Patient ID
        - HADM_ID: Hospital admission ID
        - ITEMID: Lab test ID (foreign key to D_LABITEMS)
        - CHARTTIME: When the lab was charted
        - VALUE: Lab result value (string, may contain ranges)
        - VALUENUM: Numeric lab result
        - VALUEUOM: Unit of measurement

        Args:
            chunksize: If provided, return iterator yielding chunks

        Returns:
            DataFrame or iterator
        """
        if self.source == "csv":
            csv_path = self.data_dir / "LABEVENTS.csv"
            if not csv_path.exists():
                csv_path = self.data_dir / "labevents.csv"

            if chunksize:
                # Return iterator for memory efficiency
                return pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
            else:
                df = pd.read_csv(csv_path, low_memory=False)
                # Normalize column names to uppercase
                df.columns = df.columns.str.upper()

        else:  # postgres
            df = self.load_table("LABEVENTS")

        # Convert charttime
        if 'CHARTTIME' in df.columns:
            df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce')

        if not chunksize:
            logging.info(f"Loaded LABEVENTS: {len(df)} rows")

        return df

    def load_d_labitems(self) -> pd.DataFrame:
        """
        Load D_LABITEMS dictionary.

        Contains:
        - ITEMID: Lab test ID
        - LABEL: Lab test name (e.g., "Glucose", "Hemoglobin")
        - FLUID: Specimen type (Blood, Urine, etc.)
        - CATEGORY: Lab category (Chemistry, Hematology, etc.)
        """
        return self.load_table("D_LABITEMS")

    def load_diagnoses_icd(self) -> pd.DataFrame:
        """
        Load DIAGNOSES_ICD table.

        Contains:
        - SUBJECT_ID: Patient ID
        - HADM_ID: Hospital admission ID
        - ICD9_CODE: Diagnosis code (e.g., "428.0" for CHF)
        - SEQ_NUM: Sequence number (1 = primary diagnosis)
        """
        return self.load_table("DIAGNOSES_ICD")

    def load_d_icd_diagnoses(self) -> pd.DataFrame:
        """
        Load D_ICD_DIAGNOSES dictionary.

        Contains:
        - ICD9_CODE: Diagnosis code
        - SHORT_TITLE: Short description
        - LONG_TITLE: Full description
        """
        return self.load_table("D_ICD_DIAGNOSES")

    def load_prescriptions(self) -> pd.DataFrame:
        """
        Load PRESCRIPTIONS table.

        Contains:
        - SUBJECT_ID: Patient ID
        - HADM_ID: Hospital admission ID
        - DRUG: Medication name
        - DRUG_TYPE: MAIN, BASE, ADDITIVE
        - FORMULARY_DRUG_CD: Formulary code
        - STARTDATE: Start date
        - ENDDATE: End date
        """
        df = self.load_table("PRESCRIPTIONS")

        # Convert dates
        date_cols = ['STARTDATE', 'ENDDATE']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df


# ============================================================================
# Cohort Selection
# ============================================================================

def select_cohort(
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    icustays: pd.DataFrame,
    age_min: int = 18,
    age_max: Optional[int] = None,
    use_first_icu_only: bool = True,
    subject_limit: Optional[int] = None,
    min_los_hours: Optional[float] = None,
    exclude_deaths: bool = False
) -> pd.DataFrame:
    """
    Build patient cohort based on inclusion/exclusion criteria.

    Args:
        patients: PATIENTS DataFrame
        admissions: ADMISSIONS DataFrame
        icustays: ICUSTAYS DataFrame
        age_min: Minimum age in years
        age_max: Maximum age in years (None = no limit)
        use_first_icu_only: Keep only first ICU stay per patient
        subject_limit: Limit number of patients (for demo/testing)
        min_los_hours: Minimum length of stay in hours
        exclude_deaths: Exclude patients who died in hospital

    Returns:
        DataFrame with one row per patient-ICU stay, containing:
        - SUBJECT_ID, HADM_ID, ICUSTAY_ID
        - AGE (at admission)
        - GENDER
        - ETHNICITY
        - LOS (length of stay in days)

    Rationale:
        Cohort selection affects:
        1. Data quality: Very short stays may have incomplete labs
        2. Independence: Multiple ICU stays per patient can leak information
        3. Clinical relevance: Focus on adult patients
        4. Development speed: Small cohort for quick iteration
    """
    logging.info("Starting cohort selection...")

    # Merge tables to get complete patient-admission-ICU information
    cohort = icustays.merge(
        admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ETHNICITY', 'HOSPITAL_EXPIRE_FLAG']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    cohort = cohort.merge(
        patients[['SUBJECT_ID', 'GENDER', 'DOB']],
        on='SUBJECT_ID',
        how='inner'
    )

    logging.info(f"After initial merge: {len(cohort)} ICU stays")

    # Convert date columns to datetime
    cohort['ADMITTIME'] = pd.to_datetime(cohort['ADMITTIME'], errors='coerce')
    cohort['DOB'] = pd.to_datetime(cohort['DOB'], errors='coerce')

    # Calculate age at admission (handle overflow from obfuscated dates)
    # Use year difference instead of day difference to avoid overflow
    cohort['AGE'] = cohort['ADMITTIME'].dt.year - cohort['DOB'].dt.year

    # Adjust for birthdate not yet reached in admission year
    # Subtract 1 if birthday hasn't occurred yet
    not_birthday_yet = (
        (cohort['ADMITTIME'].dt.month < cohort['DOB'].dt.month) |
        ((cohort['ADMITTIME'].dt.month == cohort['DOB'].dt.month) &
         (cohort['ADMITTIME'].dt.day < cohort['DOB'].dt.day))
    )
    cohort.loc[not_birthday_yet, 'AGE'] -= 1

    # MIMIC-III obfuscates ages >89 as 300+, remap to 91.4 (median of 90-93)
    cohort.loc[cohort['AGE'] > 89, 'AGE'] = 91.4

    # Filter by age
    cohort = cohort[cohort['AGE'] >= age_min]
    logging.info(f"After age >= {age_min} filter: {len(cohort)} ICU stays")

    if age_max is not None:
        cohort = cohort[cohort['AGE'] <= age_max]
        logging.info(f"After age <= {age_max} filter: {len(cohort)} ICU stays")

    # Filter by length of stay
    if min_los_hours is not None:
        min_los_days = min_los_hours / 24
        cohort = cohort[cohort['LOS'] >= min_los_days]
        logging.info(f"After LOS >= {min_los_hours}h filter: {len(cohort)} ICU stays")

    # Exclude deaths if requested
    if exclude_deaths:
        cohort = cohort[cohort['HOSPITAL_EXPIRE_FLAG'] == 0]
        logging.info(f"After excluding deaths: {len(cohort)} ICU stays")

    # Keep only first ICU stay per patient
    if use_first_icu_only:
        cohort = cohort.sort_values(['SUBJECT_ID', 'INTIME'])
        cohort = cohort.groupby('SUBJECT_ID').first().reset_index()
        logging.info(f"After keeping first ICU stay only: {len(cohort)} patients")

    # Limit number of subjects for development
    if subject_limit is not None:
        cohort = cohort.head(subject_limit)
        logging.info(f"Limited to {subject_limit} subjects")

    # Select and order final columns
    final_cols = [
        'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
        'AGE', 'GENDER', 'ETHNICITY',
        'INTIME', 'OUTTIME', 'LOS'
    ]

    cohort = cohort[final_cols]

    logging.info(f"Final cohort: {len(cohort)} patients")
    logging.info(f"Age range: {cohort['AGE'].min():.1f} - {cohort['AGE'].max():.1f}")
    logging.info(f"Gender distribution:\n{cohort['GENDER'].value_counts()}")

    return cohort


# ============================================================================
# Lab Events Filtering
# ============================================================================

def filter_labs_for_cohort(
    labevents: pd.DataFrame,
    cohort: pd.DataFrame,
    d_labitems: pd.DataFrame,
    top_k: Optional[int] = None,
    min_patient_count: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter lab events for the selected cohort and top-K most common labs.

    Args:
        labevents: Full LABEVENTS DataFrame
        cohort: Selected cohort DataFrame
        d_labitems: Lab items dictionary
        top_k: Keep only top-K most frequent labs (None = keep all)
        min_patient_count: Minimum number of patients with this lab

    Returns:
        Tuple of (filtered_labevents, selected_labitems)

    Rationale:
        MIMIC-III has 700+ different lab tests. Many are:
        - Rare (ordered for <10 patients)
        - Redundant (multiple codes for similar tests)
        - Not informative (qualitative results)

        Focusing on top-K ensures:
        - Statistical power (enough samples per lab)
        - Clinical relevance (commonly ordered labs)
        - Computational efficiency (smaller graph)
    """
    logging.info("Filtering lab events for cohort...")

    # Keep only labs for patients in cohort
    cohort_subject_ids = set(cohort['SUBJECT_ID'])
    labs = labevents[labevents['SUBJECT_ID'].isin(cohort_subject_ids)].copy()

    logging.info(f"Labs for cohort patients: {len(labs)} events")

    # Keep only numeric values (VALUENUM not null)
    labs = labs[labs['VALUENUM'].notna()]
    logging.info(f"After keeping numeric values: {len(labs)} events")

    # Count frequency of each lab test
    lab_counts = labs.groupby('ITEMID').agg({
        'SUBJECT_ID': 'nunique',  # Number of unique patients
        'VALUENUM': 'count'  # Total number of measurements
    }).rename(columns={'SUBJECT_ID': 'NUM_PATIENTS', 'VALUENUM': 'NUM_MEASUREMENTS'})

    # Filter by minimum patient count
    lab_counts = lab_counts[lab_counts['NUM_PATIENTS'] >= min_patient_count]

    # Keep top-K most common labs (by number of patients)
    if top_k is not None:
        lab_counts = lab_counts.nlargest(top_k, 'NUM_PATIENTS')

    logging.info(f"Selected {len(lab_counts)} lab tests")

    # Filter labs to selected ITEMIDs
    selected_itemids = set(lab_counts.index)
    labs = labs[labs['ITEMID'].isin(selected_itemids)]

    logging.info(f"Final lab events: {len(labs)}")

    # Get lab item names
    selected_labitems = d_labitems[d_labitems['ITEMID'].isin(selected_itemids)].copy()
    selected_labitems = selected_labitems.merge(
        lab_counts,
        left_on='ITEMID',
        right_index=True
    )

    logging.info(f"Lab test examples:\n{selected_labitems[['ITEMID', 'LABEL', 'NUM_PATIENTS']].head(10)}")

    return labs, selected_labitems


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    """
    Quick test of data loading functionality.

    To run: python io_mimic.py
    (Requires MIMIC-III CSV files in ../data/raw/)
    """
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Check if data directory exists
    data_dir = Path("../data/raw")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please place MIMIC-III CSV files in data/raw/")
        sys.exit(1)

    # Initialize loader
    loader = MIMICLoader(data_dir, source="csv")

    # Load core tables
    print("\n" + "="*70)
    print("Testing MIMIC-III Data Loading")
    print("="*70)

    patients = loader.load_patients()
    print(f"\n✓ PATIENTS: {len(patients)} patients")
    print(f"  Columns: {', '.join(patients.columns)}")

    admissions = loader.load_admissions()
    print(f"\n✓ ADMISSIONS: {len(admissions)} admissions")

    icustays = loader.load_icustays()
    print(f"\n✓ ICUSTAYS: {len(icustays)} ICU stays")

    # Test cohort selection
    cohort = select_cohort(
        patients, admissions, icustays,
        age_min=18,
        use_first_icu_only=True,
        subject_limit=100  # Small sample for testing
    )

    print(f"\n✓ COHORT: {len(cohort)} patients selected")
    print(f"  Age: {cohort['AGE'].mean():.1f} ± {cohort['AGE'].std():.1f} years")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
