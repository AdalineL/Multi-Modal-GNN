"""
Data Preprocessing and Feature Engineering Module

This module transforms raw MIMIC-III data into features suitable for GNN training:
1. Lab value aggregation and normalization
2. Diagnosis code processing (ICD-9 collapse to 3-digit)
3. Medication name normalization
4. Patient demographic features

All processing is config-driven and produces intermediate parquet files.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

from utils import LabNormalizer, remove_outliers, save_dataframe


# ============================================================================
# Lab Value Processing
# ============================================================================

def aggregate_lab_values(
    labs: pd.DataFrame,
    cohort: pd.DataFrame,
    method: str = "last",
    remove_outliers_flag: bool = True,
    outlier_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Aggregate multiple lab measurements per patient-lab pair into single value.

    Args:
        labs: Lab events DataFrame (SUBJECT_ID, ITEMID, VALUENUM, CHARTTIME)
        cohort: Patient cohort DataFrame
        method: Aggregation method ("last", "mean", "median", "min", "max")
        remove_outliers_flag: Whether to remove outliers before aggregation
        outlier_threshold: Threshold for outlier removal (std deviations)

    Returns:
        DataFrame with columns [SUBJECT_ID, ITEMID, VALUE]

    Rationale:
        Patients may have the same lab test multiple times during ICU stay.
        We need a single representative value per patient-lab pair.

        Options:
        - "last": Most recent value reflects current state (CHOSEN)
        - "mean": Average across stay
        - "median": Robust to outliers
        - "min"/"max": Extreme values may indicate severity
    """
    logging.info(f"Aggregating lab values using method: {method}")

    # Ensure we only process labs for cohort patients
    cohort_ids = set(cohort['SUBJECT_ID'])
    labs = labs[labs['SUBJECT_ID'].isin(cohort_ids)].copy()

    # Remove outliers per lab test if requested
    if remove_outliers_flag:
        logging.info("Removing outliers from lab values...")
        cleaned_labs = []

        for itemid, group in labs.groupby('ITEMID'):
            clean_values = remove_outliers(
                group['VALUENUM'],
                method='std',
                threshold=outlier_threshold
            )
            group = group.copy()
            group['VALUENUM'] = clean_values
            cleaned_labs.append(group)

        labs = pd.concat(cleaned_labs, ignore_index=True)

        # Drop rows with NaN after outlier removal
        labs = labs[labs['VALUENUM'].notna()]

    # Aggregation strategies
    if method == "last":
        # Sort by charttime and take last value per patient-lab
        labs = labs.sort_values(['SUBJECT_ID', 'ITEMID', 'CHARTTIME'])
        labs_agg = labs.groupby(['SUBJECT_ID', 'ITEMID']).tail(1)
        labs_agg = labs_agg[['SUBJECT_ID', 'ITEMID', 'VALUENUM']].copy()

    elif method in ["mean", "median", "min", "max"]:
        # Statistical aggregation
        agg_func = {
            'mean': 'mean',
            'median': 'median',
            'min': 'min',
            'max': 'max'
        }[method]

        labs_agg = labs.groupby(['SUBJECT_ID', 'ITEMID'])['VALUENUM'].agg(agg_func).reset_index()

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    labs_agg.rename(columns={'VALUENUM': 'VALUE'}, inplace=True)

    logging.info(f"Aggregated to {len(labs_agg)} patient-lab pairs")
    logging.info(f"Coverage: {labs_agg['SUBJECT_ID'].nunique()} patients, {labs_agg['ITEMID'].nunique()} labs")

    return labs_agg


def normalize_lab_values(
    labs_agg: pd.DataFrame,
    method: str = "zscore"
) -> Tuple[pd.DataFrame, LabNormalizer]:
    """
    Normalize lab values per lab test.

    Args:
        labs_agg: Aggregated lab values (SUBJECT_ID, ITEMID, VALUE)
        method: Normalization method ("zscore", "minmax", "robust")

    Returns:
        Tuple of (normalized DataFrame, fitted normalizer object)

    Rationale:
        Different lab tests have vastly different scales:
        - Glucose: 70-200 mg/dL
        - WBC: 4-11 K/uL
        - Creatinine: 0.5-1.5 mg/dL

        Without normalization, high-magnitude labs dominate the model.
        Z-score normalization makes all labs comparable.
    """
    logging.info(f"Normalizing lab values using method: {method}")

    normalizer = LabNormalizer(method=method)
    labs_normalized = []

    for itemid, group in labs_agg.groupby('ITEMID'):
        group = group.copy()
        normalized_values = normalizer.fit_transform(group['VALUE'], str(itemid))
        group['VALUE_NORMALIZED'] = normalized_values
        labs_normalized.append(group)

    labs_agg_norm = pd.concat(labs_normalized, ignore_index=True)

    # Drop any remaining NaN
    labs_agg_norm = labs_agg_norm[labs_agg_norm['VALUE_NORMALIZED'].notna()]

    # Ensure SUBJECT_ID is integer (concat can convert to float)
    labs_agg_norm['SUBJECT_ID'] = labs_agg_norm['SUBJECT_ID'].astype('int64')

    # Try to convert ITEMID to int if possible (MIMIC-III), otherwise keep as string (eICU)
    try:
        labs_agg_norm['ITEMID'] = labs_agg_norm['ITEMID'].astype('int64')
    except (ValueError, TypeError):
        # ITEMID is string (e.g., eICU lab names), keep as is
        pass

    logging.info(f"Normalized {len(labs_agg_norm)} lab values")

    return labs_agg_norm, normalizer


# ============================================================================
# Diagnosis Processing
# ============================================================================

def process_diagnoses(
    diagnoses: pd.DataFrame,
    cohort: pd.DataFrame,
    collapse_to_3digit: bool = True,
    top_k: Optional[int] = None,
    min_patient_count: int = 5
) -> pd.DataFrame:
    """
    Process ICD-9 diagnosis codes for graph construction.

    Args:
        diagnoses: DIAGNOSES_ICD DataFrame (SUBJECT_ID, HADM_ID, ICD9_CODE)
        cohort: Patient cohort DataFrame
        collapse_to_3digit: Collapse codes to 3-digit categories
        top_k: Keep only top-K most frequent diagnoses
        min_patient_count: Minimum patients per diagnosis

    Returns:
        DataFrame with [SUBJECT_ID, ICD9_CODE, ICD3_CODE]

    Rationale:
        ICD-9 codes are hierarchical:
        - 428: Heart failure (broad category)
        - 428.0: Congestive heart failure, unspecified
        - 428.1: Left heart failure
        - 428.9: Heart failure, unspecified

        Collapsing to 3-digit:
        - Reduces sparsity (more patients per code)
        - Maintains clinical meaning (disease category)
        - Standard practice in EHR research
    """
    logging.info("Processing diagnosis codes...")

    # Keep only diagnoses for cohort patients
    cohort_hadm_ids = set(cohort['HADM_ID'])
    dx = diagnoses[diagnoses['HADM_ID'].isin(cohort_hadm_ids)].copy()

    logging.info(f"Diagnoses for cohort: {len(dx)} records")

    # Clean ICD-9 codes (remove whitespace, convert to string)
    dx['ICD9_CODE'] = dx['ICD9_CODE'].astype(str).str.strip()

    # Remove invalid codes
    dx = dx[dx['ICD9_CODE'] != '']
    dx = dx[dx['ICD9_CODE'].notna()]

    # Collapse to 3-digit codes if requested
    if collapse_to_3digit:
        # Take first 3 characters
        dx['ICD3_CODE'] = dx['ICD9_CODE'].str[:3]
        diagnosis_col = 'ICD3_CODE'
        logging.info(f"Collapsed to 3-digit codes: {dx['ICD3_CODE'].nunique()} unique codes")
    else:
        dx['ICD3_CODE'] = dx['ICD9_CODE']
        diagnosis_col = 'ICD9_CODE'

    # Get patient-level diagnoses (one row per patient-diagnosis pair)
    # Filter to only cohort patients (diagnoses already has SUBJECT_ID)
    cohort_ids = set(cohort['SUBJECT_ID'])
    dx = dx[dx['SUBJECT_ID'].isin(cohort_ids)].copy()

    # Keep unique patient-diagnosis pairs along with metadata if available
    cols_to_keep = ['SUBJECT_ID', diagnosis_col]
    if 'DIAGNOSIS_CATEGORY' in dx.columns:
        cols_to_keep.append('DIAGNOSIS_CATEGORY')
    if 'DIAGNOSIS_SUBCATEGORY' in dx.columns:
        cols_to_keep.append('DIAGNOSIS_SUBCATEGORY')
    if 'DIAGNOSIS_PRIORITY' in dx.columns:
        cols_to_keep.append('DIAGNOSIS_PRIORITY')

    dx_patient = dx[cols_to_keep].drop_duplicates(subset=['SUBJECT_ID', diagnosis_col])

    # Count frequency of each diagnosis
    dx_counts = dx_patient[diagnosis_col].value_counts()

    # Filter by minimum patient count
    dx_counts = dx_counts[dx_counts >= min_patient_count]

    # Keep top-K if specified
    if top_k is not None:
        dx_counts = dx_counts.head(top_k)

    logging.info(f"Selected {len(dx_counts)} diagnoses")

    # Filter to selected diagnoses
    selected_codes = set(dx_counts.index)
    dx_patient = dx_patient[dx_patient[diagnosis_col].isin(selected_codes)]

    logging.info(f"Final: {len(dx_patient)} patient-diagnosis pairs")
    logging.info(f"Coverage: {dx_patient['SUBJECT_ID'].nunique()} patients")

    # Log most common diagnoses
    logging.info(f"Top 10 diagnoses:\n{dx_counts.head(10)}")

    return dx_patient


# ============================================================================
# Medication Processing
# ============================================================================

def normalize_drug_name(drug: str) -> str:
    """
    Normalize medication names to generic forms.

    Args:
        drug: Raw drug name from prescriptions

    Returns:
        Normalized drug name

    Rationale:
        MIMIC-III drug names are inconsistent:
        - "Aspirin 81mg", "Aspirin EC 81mg", "ASA 81mg" → "aspirin"
        - "Metoprolol Tartrate 25mg", "Metoprolol 50mg" → "metoprolol"

        Normalization consolidates variations.
    """
    if pd.isna(drug):
        return ""

    # Convert to lowercase
    drug = str(drug).lower()

    # Remove dosage information (e.g., "50mg", "10 mg")
    drug = re.sub(r'\d+\.?\d*\s*(mg|mcg|ml|g|%|units?)', '', drug)

    # Remove common suffixes/prefixes
    drug = re.sub(r'\b(tablet|capsule|injection|solution|suspension|syrup|cream|ointment)\b', '', drug)
    drug = re.sub(r'\b(oral|topical|iv|intravenous|subcutaneous)\b', '', drug)

    # Remove special characters and extra whitespace
    drug = re.sub(r'[^\w\s]', ' ', drug)
    drug = re.sub(r'\s+', ' ', drug).strip()

    # Extract first word (often the generic name)
    words = drug.split()
    if len(words) > 0:
        drug = words[0]

    return drug


def process_medications(
    prescriptions: pd.DataFrame,
    cohort: pd.DataFrame,
    normalize_names: bool = True,
    top_k: Optional[int] = None,
    min_patient_count: int = 5
) -> pd.DataFrame:
    """
    Process medication prescriptions for graph construction.

    Args:
        prescriptions: PRESCRIPTIONS DataFrame
        cohort: Patient cohort DataFrame
        normalize_names: Apply drug name normalization
        top_k: Keep only top-K most frequent medications
        min_patient_count: Minimum patients per medication

    Returns:
        DataFrame with [SUBJECT_ID, DRUG]

    Rationale:
        Medication data is noisy:
        - Same drug with different names (brand vs generic)
        - Different dosages and formulations
        - Inconsistent capitalization

        Processing steps:
        1. Normalize names to generic forms
        2. Keep only common medications (top-K)
        3. Create patient-medication edges
    """
    logging.info("Processing medications...")

    # Keep only prescriptions for cohort patients
    cohort_hadm_ids = set(cohort['HADM_ID'])
    meds = prescriptions[prescriptions['HADM_ID'].isin(cohort_hadm_ids)].copy()

    logging.info(f"Prescriptions for cohort: {len(meds)} records")

    # Clean drug names
    meds['DRUG'] = meds['DRUG'].astype(str).str.strip()
    meds = meds[meds['DRUG'] != '']
    meds = meds[meds['DRUG'].notna()]

    # Normalize drug names if requested
    if normalize_names:
        logging.info("Normalizing drug names...")
        meds['DRUG_NORMALIZED'] = meds['DRUG'].apply(normalize_drug_name)
        # Remove empty normalized names
        meds = meds[meds['DRUG_NORMALIZED'] != '']
        drug_col = 'DRUG_NORMALIZED'
    else:
        drug_col = 'DRUG'

    # Get patient-level medications
    # Filter to only cohort patients (prescriptions already has SUBJECT_ID)
    cohort_ids = set(cohort['SUBJECT_ID'])
    meds = meds[meds['SUBJECT_ID'].isin(cohort_ids)].copy()

    # Extract and normalize dosage information if available
    if 'DOSAGE' in meds.columns:
        logging.info("Processing medication dosages...")
        # Extract numeric dosage (handles formats like "5 3" → 5.0, "10.5" → 10.5)
        meds['DOSAGE_STR'] = meds['DOSAGE'].astype(str)
        meds['DOSAGE_CLEAN'] = meds['DOSAGE_STR'].str.extract(r'(\d+\.?\d*)')[0]
        meds['DOSAGE_CLEAN'] = pd.to_numeric(meds['DOSAGE_CLEAN'], errors='coerce')

        # Count how many have valid dosages
        has_dosage = meds['DOSAGE_CLEAN'].notna().sum()
        total = len(meds)
        logging.info(f"  Extracted dosages for {has_dosage}/{total} ({100*has_dosage/total:.1f}%) medications")

        # Normalize dosage per drug (z-score within each medication)
        # This accounts for different drugs having different dose ranges
        # e.g., Warfarin (2.5-10mg) vs Aspirin (81-325mg)
        def safe_normalize(x):
            """Z-score normalization with fallback for single-value groups"""
            if len(x) <= 1 or x.std() < 1e-8:
                return pd.Series(0.0, index=x.index)  # Single value or no variance
            return (x - x.mean()) / x.std()

        meds['DOSAGE_NORM'] = meds.groupby(drug_col)['DOSAGE_CLEAN'].transform(safe_normalize)

        # Fill NaN dosages with 0 (neutral weight)
        meds['DOSAGE_NORM'] = meds['DOSAGE_NORM'].fillna(0.0)

        logging.info(f"  Normalized dosage: mean={meds['DOSAGE_NORM'].mean():.3f}, std={meds['DOSAGE_NORM'].std():.3f}")

    # Keep unique patient-medication pairs along with metadata if available
    cols_to_keep = ['SUBJECT_ID', drug_col]
    if 'ROUTE' in meds.columns:
        cols_to_keep.append('ROUTE')
    if 'FREQUENCY' in meds.columns:
        cols_to_keep.append('FREQUENCY')
    if 'PRN' in meds.columns:
        cols_to_keep.append('PRN')
    if 'IV_ADMIXTURE' in meds.columns:
        cols_to_keep.append('IV_ADMIXTURE')
    if 'DOSAGE_NORM' in meds.columns:
        cols_to_keep.append('DOSAGE_NORM')
        cols_to_keep.append('DOSAGE_CLEAN')  # Keep raw dosage too for debugging

    # Aggregate patient-medication pairs (take mean dosage if multiple records)
    if 'DOSAGE_NORM' in cols_to_keep:
        # Group by patient-drug and aggregate dosages
        agg_dict = {}
        for col in cols_to_keep:
            if col in ['DOSAGE_NORM', 'DOSAGE_CLEAN']:
                agg_dict[col] = 'mean'  # Average dosage
            elif col in ['SUBJECT_ID', drug_col]:
                agg_dict[col] = 'first'
            else:
                agg_dict[col] = 'first'  # Take first value for categorical

        meds_patient = meds[cols_to_keep].groupby(['SUBJECT_ID', drug_col], as_index=False).agg(agg_dict)
    else:
        meds_patient = meds[cols_to_keep].drop_duplicates(subset=['SUBJECT_ID', drug_col])

    # Count frequency of each medication
    med_counts = meds_patient[drug_col].value_counts()

    # Filter by minimum patient count
    med_counts = med_counts[med_counts >= min_patient_count]

    # Keep top-K if specified
    if top_k is not None:
        med_counts = med_counts.head(top_k)

    logging.info(f"Selected {len(med_counts)} medications")

    # Filter to selected medications
    selected_drugs = set(med_counts.index)
    meds_patient = meds_patient[meds_patient[drug_col].isin(selected_drugs)]

    # Rename column for consistency
    meds_patient.rename(columns={drug_col: 'DRUG'}, inplace=True)

    logging.info(f"Final: {len(meds_patient)} patient-medication pairs")
    logging.info(f"Coverage: {meds_patient['SUBJECT_ID'].nunique()} patients")

    # Log dosage statistics if available
    if 'DOSAGE_NORM' in meds_patient.columns:
        dosage_available = meds_patient['DOSAGE_NORM'].ne(0).sum()
        logging.info(f"Dosage info: {dosage_available}/{len(meds_patient)} ({100*dosage_available/len(meds_patient):.1f}%) pairs have dosage")

    # Log most common medications
    logging.info(f"Top 10 medications:\n{med_counts.head(10)}")

    return meds_patient


# ============================================================================
# Patient Demographics Features
# ============================================================================

def create_demographic_features(
    cohort: pd.DataFrame,
    apache_scores: Optional[pd.DataFrame] = None,
    include_age: bool = True,
    include_gender: bool = True,
    include_ethnicity: bool = False
) -> pd.DataFrame:
    """
    Create enriched patient demographic features for node embeddings.

    Args:
        cohort: Patient cohort DataFrame
        apache_scores: APACHE severity scores (eICU only)
        include_age: Include age as continuous feature
        include_gender: Include gender as one-hot feature
        include_ethnicity: Include ethnicity as one-hot feature

    Returns:
        DataFrame with [SUBJECT_ID, feature_1, feature_2, ...]

    Features (Priority 1):
        - Demographics: age, gender, ethnicity, height, weight
        - APACHE scores: acute physiology score, APACHE score, mortality predictions
        - Care context: unit type, admit source, stay type
    """
    logging.info("Creating enriched demographic features...")

    features = cohort[['SUBJECT_ID']].copy()

    # Age feature
    if include_age:
        # Normalize age to [0, 1] range (assuming max age ~100)
        features['AGE_NORMALIZED'] = cohort['AGE'] / 100.0
        logging.info(f"Added age feature (mean: {cohort['AGE'].mean():.1f} years)")

    # Gender one-hot encoding
    if include_gender:
        gender_dummies = pd.get_dummies(cohort['GENDER'], prefix='GENDER')
        features = pd.concat([features, gender_dummies], axis=1)
        logging.info(f"Added gender features: {list(gender_dummies.columns)}")

    # Ethnicity one-hot encoding (if available and requested)
    if include_ethnicity and 'ETHNICITY' in cohort.columns:
        # Group rare ethnicities into "OTHER"
        ethnicity = cohort['ETHNICITY'].copy()
        ethnicity_counts = ethnicity.value_counts()
        rare_ethnicities = ethnicity_counts[ethnicity_counts < 10].index
        ethnicity = ethnicity.replace(rare_ethnicities, 'OTHER')

        ethnicity_dummies = pd.get_dummies(ethnicity, prefix='ETHNICITY')
        features = pd.concat([features, ethnicity_dummies], axis=1)
        logging.info(f"Added ethnicity features: {len(ethnicity_dummies.columns)} categories")

    # Height and Weight (eICU)
    if 'ADMISSION_HEIGHT' in cohort.columns:
        # Normalize height (typical range: 120-200 cm)
        features['HEIGHT_NORMALIZED'] = cohort['ADMISSION_HEIGHT'].fillna(
            cohort['ADMISSION_HEIGHT'].median()
        ) / 200.0
        logging.info("Added height feature")

    if 'ADMISSION_WEIGHT' in cohort.columns:
        # Normalize weight (typical range: 30-200 kg)
        features['WEIGHT_NORMALIZED'] = cohort['ADMISSION_WEIGHT'].fillna(
            cohort['ADMISSION_WEIGHT'].median()
        ) / 200.0
        logging.info("Added weight feature")

    # APACHE Scores (eICU Priority 1 features)
    if apache_scores is not None:
        # Merge APACHE scores
        features = features.merge(
            apache_scores[['SUBJECT_ID', 'acutephysiologyscore', 'apachescore',
                          'predictedicumortality', 'predictedhospitalmortality']],
            on='SUBJECT_ID',
            how='left'
        )

        # Normalize APACHE scores (range 0-299)
        if 'acutephysiologyscore' in features.columns:
            features['APACHE_APS_NORM'] = features['acutephysiologyscore'].fillna(0) / 299.0
            features = features.drop(columns=['acutephysiologyscore'])
            logging.info("Added APACHE Acute Physiology Score")

        if 'apachescore' in features.columns:
            features['APACHE_SCORE_NORM'] = features['apachescore'].fillna(0) / 299.0
            features = features.drop(columns=['apachescore'])
            logging.info("Added APACHE Score")

        # Mortality predictions (already 0-1 probabilities)
        if 'predictedicumortality' in features.columns:
            features['PRED_ICU_MORTALITY'] = features['predictedicumortality'].fillna(0)
            features = features.drop(columns=['predictedicumortality'])
            logging.info("Added Predicted ICU Mortality")

        if 'predictedhospitalmortality' in features.columns:
            features['PRED_HOSP_MORTALITY'] = features['predictedhospitalmortality'].fillna(0)
            features = features.drop(columns=['predictedhospitalmortality'])
            logging.info("Added Predicted Hospital Mortality")

    # Care Context (eICU)
    if 'UNIT_TYPE' in cohort.columns:
        # Group rare unit types
        unit_type = cohort['UNIT_TYPE'].copy()
        unit_counts = unit_type.value_counts()
        top_units = unit_counts.head(5).index
        unit_type = unit_type.apply(lambda x: x if x in top_units else 'Other')

        unit_dummies = pd.get_dummies(unit_type, prefix='UNIT')
        features = pd.concat([features, unit_dummies], axis=1)
        logging.info(f"Added unit type features: {len(unit_dummies.columns)} categories")

    if 'UNIT_ADMIT_SOURCE' in cohort.columns:
        # Group rare admit sources
        admit_source = cohort['UNIT_ADMIT_SOURCE'].copy()
        source_counts = admit_source.value_counts()
        top_sources = source_counts.head(5).index
        admit_source = admit_source.apply(lambda x: x if x in top_sources else 'Other')

        source_dummies = pd.get_dummies(admit_source, prefix='ADMIT_SRC')
        features = pd.concat([features, source_dummies], axis=1)
        logging.info(f"Added admit source features: {len(source_dummies.columns)} categories")

    # Number of feature columns (excluding SUBJECT_ID)
    num_features = len(features.columns) - 1
    logging.info(f"Total patient features: {num_features}")

    return features


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def preprocess_pipeline(
    loader,
    cohort: pd.DataFrame,
    config: Dict,
    output_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Run complete preprocessing pipeline and save intermediate files.

    Args:
        loader: MIMICLoader instance
        cohort: Selected patient cohort
        config: Configuration dictionary
        output_dir: Where to save intermediate files

    Returns:
        Dictionary containing all processed DataFrames

    This is the main entry point for preprocessing. It orchestrates:
    1. Lab value aggregation and normalization
    2. Diagnosis code processing
    3. Medication processing
    4. Demographic feature creation
    5. Saving all intermediate files
    """
    logging.info("="*70)
    logging.info("Starting preprocessing pipeline")
    logging.info("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Save cohort
    save_dataframe(cohort, output_dir / "cohort.parquet")
    results['cohort'] = cohort

    # ------------------------------------------------------------------------
    # Process Labs
    # ------------------------------------------------------------------------
    logging.info("\n" + "-"*70)
    logging.info("PROCESSING LAB VALUES")
    logging.info("-"*70)

    # Load raw lab events
    d_labitems = loader.load_d_labitems()
    labevents = loader.load_labevents()

    # Filter to top-K labs
    from io_mimic import filter_labs_for_cohort
    labs_filtered, selected_labitems = filter_labs_for_cohort(
        labevents,
        cohort,
        d_labitems,
        top_k=config['feature_space']['labs']['top_k'],
        min_patient_count=config['feature_space']['labs'].get('min_patient_count', 10)
    )

    # Aggregate lab values
    labs_agg = aggregate_lab_values(
        labs_filtered,
        cohort,
        method=config['feature_space']['labs']['aggregate'],
        remove_outliers_flag=config['feature_space']['labs'].get('outlier_std_threshold') is not None,
        outlier_threshold=config['feature_space']['labs'].get('outlier_std_threshold', 5.0)
    )

    # Normalize lab values
    labs_normalized, normalizer = normalize_lab_values(
        labs_agg,
        method=config['feature_space']['labs']['normalize']
    )

    # Save
    save_dataframe(selected_labitems, output_dir / "labitems.parquet")
    save_dataframe(labs_normalized, output_dir / "labs_normalized.parquet")

    results['labitems'] = selected_labitems
    results['labs'] = labs_normalized
    results['lab_normalizer'] = normalizer

    # ------------------------------------------------------------------------
    # Process Diagnoses
    # ------------------------------------------------------------------------
    logging.info("\n" + "-"*70)
    logging.info("PROCESSING DIAGNOSES")
    logging.info("-"*70)

    diagnoses_icd = loader.load_diagnoses_icd()

    diagnoses_processed = process_diagnoses(
        diagnoses_icd,
        cohort,
        collapse_to_3digit=config['feature_space']['diagnoses']['collapse_to_3digit'],
        top_k=config['feature_space']['diagnoses']['top_k'],
        min_patient_count=config['feature_space']['diagnoses'].get('min_patient_count', 5)
    )

    save_dataframe(diagnoses_processed, output_dir / "diagnoses.parquet")
    results['diagnoses'] = diagnoses_processed

    # ------------------------------------------------------------------------
    # Process Medications
    # ------------------------------------------------------------------------
    logging.info("\n" + "-"*70)
    logging.info("PROCESSING MEDICATIONS")
    logging.info("-"*70)

    prescriptions = loader.load_prescriptions()

    medications_processed = process_medications(
        prescriptions,
        cohort,
        normalize_names=config['feature_space']['medications']['normalize_names'],
        top_k=config['feature_space']['medications']['top_k'],
        min_patient_count=config['feature_space']['medications'].get('min_patient_count', 5)
    )

    save_dataframe(medications_processed, output_dir / "medications.parquet")
    results['medications'] = medications_processed

    # ------------------------------------------------------------------------
    # Load APACHE Scores (eICU only)
    # ------------------------------------------------------------------------
    apache_scores = None
    if hasattr(loader, 'load_apache_for_cohort'):
        logging.info("\n" + "-"*70)
        logging.info("LOADING APACHE SCORES")
        logging.info("-"*70)
        try:
            apache_scores = loader.load_apache_for_cohort(cohort)
            logging.info(f"Loaded APACHE scores for {len(apache_scores)} patients")
            logging.info(f"APACHE features: {apache_scores.columns.tolist()}")
        except Exception as e:
            logging.warning(f"Could not load APACHE scores: {e}")
            apache_scores = None

    # ------------------------------------------------------------------------
    # Create Demographics Features
    # ------------------------------------------------------------------------
    logging.info("\n" + "-"*70)
    logging.info("CREATING DEMOGRAPHIC FEATURES")
    logging.info("-"*70)

    demographics = create_demographic_features(
        cohort,
        apache_scores=apache_scores,
        include_age=config['feature_space']['demographics']['include_age'],
        include_gender=config['feature_space']['demographics']['include_gender'],
        include_ethnicity=config['feature_space']['demographics']['include_ethnicity']
    )

    save_dataframe(demographics, output_dir / "demographics.parquet")
    results['demographics'] = demographics

    # ------------------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------------------
    logging.info("\n" + "="*70)
    logging.info("PREPROCESSING COMPLETE - SUMMARY")
    logging.info("="*70)
    logging.info(f"Cohort: {len(cohort)} patients")
    logging.info(f"Labs: {labs_normalized['ITEMID'].nunique()} unique tests, {len(labs_normalized)} measurements")
    logging.info(f"Diagnoses: {diagnoses_processed['ICD3_CODE'].nunique()} unique codes, {len(diagnoses_processed)} patient-diagnosis pairs")
    logging.info(f"Medications: {medications_processed['DRUG'].nunique()} unique drugs, {len(medications_processed)} patient-medication pairs")
    logging.info(f"Demographics: {len(demographics.columns)-1} features per patient")
    logging.info(f"All files saved to: {output_dir}")
    logging.info("="*70)

    return results


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """
    Run preprocessing pipeline from command line.

    Usage:
        python preprocess.py
    """
    import sys
    sys.path.append(str(Path(__file__).parent))

    from utils import load_config, setup_logging, set_random_seeds
    from config_helper import load_and_process_config, print_experiment_summary

    # Load configuration with experiment mode processing
    config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    config = load_and_process_config(str(config_path))

    # Setup
    setup_logging(
        level=config['logging']['level'],
        log_file=None  # Console only for now
    )

    # Print experiment configuration
    print_experiment_summary(config)

    set_random_seeds(config['train']['seed'])

    # Determine dataset type
    dataset_type = config['data'].get('dataset', 'mimic3').lower()

    if dataset_type == 'eicu':
        # Load eICU data
        from io_eicu import eICULoader, select_cohort
        logging.info("Loading eICU data...")
        loader = eICULoader(config['data']['raw_dir'])
        patients = loader.load_patients()
        cohort = select_cohort(patients, loader=loader, **config['cohort'])

    else:  # Default to MIMIC-III
        # Load MIMIC-III data
        from io_mimic import MIMICLoader, select_cohort
        logging.info("Loading MIMIC-III data...")
        loader = MIMICLoader(config['data']['raw_dir'], source='csv')
        patients = loader.load_patients()
        admissions = loader.load_admissions()
        icustays = loader.load_icustays()
        cohort = select_cohort(
            patients, admissions, icustays,
            **config['cohort']
        )

    # Run preprocessing
    output_dir = Path(config['data']['interim_dir'])
    results = preprocess_pipeline(
        loader,
        cohort,
        config,
        output_dir
    )

    logging.info("\nPreprocessing complete!")
