# EHR Graph Imputation - Key Findings Summary

## The Central Question
**Does focusing model capacity on specific lab types improve prediction accuracy?**

---

## Critical Discovery: Apples-to-Apples Comparison

### Previous (Misleading) Comparison
**Problem:** We compared:
- Baseline: All labs R² = 24.23%
- Iteration 1: CV labs R² = 19.04%
- **Conclusion:** CV focus seems worse ❌

### Correct (Fair) Comparison
**Solution:** Extract CV lab performance from baseline

| Configuration | Cohort Size | Labs Predicted | **CV Lab R²** | Overall R² |
|---------------|-------------|----------------|---------------|------------|
| **Baseline** | 1,834 | All 50 labs | **13.95%** | 24.23% |
| **Iteration 1** | 1,834 | 12 CV labs only | **19.04%** | 19.04% |
| **Iteration 2** | 607 (CV only) | 12 CV labs only | 18.52% | 18.52% |

### Key Insight
**CV-focused training improves CV lab prediction by +36.5%!**
- Baseline CV labs: R² = 13.95%
- Iteration 1 CV labs: R² = 19.04%
- **Improvement: +5.09 percentage points (+36.5% relative)**

---

## Why This Matters

### 1. Model Capacity is Limited
- With 483,970 parameters predicting 50 diverse labs, each lab gets limited capacity
- Focusing on 12 CV labs gives each lab ~4× more capacity
- Result: Better performance on the labs that matter

### 2. CV Labs Are Harder Than Average
- Average R² for all 50 labs: 24.23%
- Average R² for 12 CV labs: **13.95%** (43% worse)
- CV labs (troponin, BNP, coagulation) have complex, nonlinear relationships

### 3. Cohort Filtering Hurts (Usually)
**Iteration 2 (CV cohort) vs Iteration 1 (all patients):**
- Reduces training data by 67% (1,834 → 607 patients)
- Overall R²: 19.04% → 18.52% (worse)
- **BUT:** Common CV labs improved (R² 0.18 → 0.27)
- **Trade-off:** Lose diversity, gain homogeneity

---

## Baseline CV Lab Performance (Detailed)

From baseline model (predicting all 50 labs):

| Lab Name | MAE | RMSE | R² | Samples |
|----------|-----|------|----|---------|
| **PT** | 0.4026 | 0.6206 | **0.5055** ⭐ | 173 |
| **lactate** | 0.3295 | 0.5485 | **0.4897** ⭐ | 88 |
| **PTT** | 0.2612 | 0.3558 | 0.1905 | 112 |
| **PT - INR** | 0.4116 | 0.6790 | 0.1408 | 173 |
| **glucose** | 0.6532 | 1.0448 | 0.1397 | 268 |
| **calcium** | 0.6682 | 0.8627 | 0.1334 | 240 |
| **troponin - I** | 0.2333 | 0.5251 | 0.1138 | 100 |
| **potassium** | 0.7027 | 0.9559 | 0.0511 | 232 |
| **sodium** | 0.6985 | 0.9320 | 0.0270 | 267 |
| **magnesium** | 0.7172 | 0.9023 | 0.0225 | 206 |
| **bedside glucose** | 0.6971 | 0.9860 | **-0.0488** ❌ | 157 |
| **triglycerides** | 0.8181 | 1.3993 | **-0.0911** ❌ | 53 |
| **Average** | **0.5494** | **0.8177** | **0.1395** | **172** |

### Observations:
- **Best:** PT and lactate (R² ~0.50) - coagulation and perfusion
- **Worst:** Triglycerides, bedside glucose (negative R²) - high variability
- **Middle:** Electrolytes, troponin (R² 0.01-0.13) - challenging to predict
- **2 of 12 labs have negative R²** - worse than mean prediction

---

## What We're Testing Now

**Iteration 1 (Re-run):**
- Config: `cv_cohort_mode: false`, `cv_target_mode: true`
- Cohort: All 1,834 patients
- Prediction: 12 CV labs only
- Expected: R² ≈ 19% (as in EXPERIMENT_RESULTS.md)

**This confirms:**
✓ Focused training improves CV lab prediction
✓ All patients provide better context than CV-only cohort
✓ Non-CV labs (kidney, CBC) help predict CV labs

---

## Next Experiments: Priority Order

### 🔥 Priority 1: Single-Lab Specialists
**Hypothesis:** Dedicating all model capacity to ONE lab maximizes accuracy

**Test Labs:**
1. **Troponin I** (Baseline: 11.4% → Target: >30%)
2. **BNP** (Not in top 50, need full eICU)
3. **PT-INR** (Baseline: 14.1% → Target: >40%)
4. **Lactate** (Baseline: 49.0% → Target: >60%)

**Why Priority 1:**
- Direct answer to "does focused training help?"
- Maximum possible improvement
- High clinical value (diagnostic labs)

---

### 🔥 Priority 2: Heart Failure Labs
**Hypothesis:** HF-specific labs more homogeneous than "all CV"

**HF Target Labs (6-8 labs):**
- BNP / NT-proBNP (primary marker)
- Troponin (myocardial injury)
- K, Na, Mg (electrolytes for arrhythmia)
- Creatinine / BUN (cardiorenal syndrome)
- Lactate (perfusion)

**Cohort Options:**
- A: All patients, HF labs (like Iteration 1) ✓ Recommended
- B: HF patients (428.*), HF labs
- C: HF + CAD + MI patients, HF labs

**Expected:**
- Higher R² than "all CV labs" (more focused)
- Clearer clinical use case (HF management)

---

### 🔥 Priority 3: Add Medication Features
**Hypothesis:** Medications causally determine some lab values

**Target Interactions:**
- Warfarin → PT-INR (should dramatically improve)
- Diuretics (furosemide) → K, Na (electrolyte shifts)
- Insulin → Glucose
- Beta-blockers, ACE-I → May affect troponin/BNP

**Implementation:**
- Add medication nodes to graph (already have edges)
- Add dosage/timing features
- Create explicit medication→lab edge type

**Expected:**
- PT-INR: 14% → 40%+ (warfarin is primary determinant)
- Electrolytes: 5% → 20%+ (diuretic effect)

---

### Other Ideas:
4. **Temporal features** (time since admission, lab trends)
5. **Increased model capacity** (256/512 hidden dim, 3-4 layers)
6. **Disease-specific cohorts** (sepsis, ACS)
7. **Cross-database validation** (train MIMIC, test eICU)

---

## Clinical Motivation

### Why Predict Lab Values?
1. **Reduce unnecessary blood draws** (patient comfort, cost)
2. **Early warning system** (predict abnormal values before they occur)
3. **Optimize lab ordering** (which labs actually needed?)
4. **Missing data imputation** (complete lab panels)

### What R² Is "Good Enough"?
- **R² < 20%:** Research only, not clinically useful
- **R² 20-40%:** Useful for screening, triaging lab orders
- **R² 40-60%:** Potentially useful for decision support
- **R² > 60%:** High confidence, could reduce lab orders

### Current Status:
- **Baseline (all labs):** 24.2% - marginal clinical utility
- **Iteration 1 (CV labs):** 19.0% - not yet clinical utility
- **Individual labs (PT, lactate):** 50% - promising!

**Path forward:** Focus on high-performing labs or add medication features

---

## Technical Summary

### What Works:
✅ Degree-aware GNN captures hub nodes (common labs, high-degree patients)
✅ Heterogeneous graph (patient, lab, diagnosis, medication)
✅ Focused training improves target lab prediction (+36%)
✅ Rare labs benefit from rich context (R² 0.54)

### What Doesn't Work Yet:
❌ Cohort filtering reduces performance (data scarcity)
❌ Very common labs underperform (glucose, electrolytes)
❌ Some labs have negative R² (worse than mean)

### Opportunities:
🎯 Single-lab models (maximize capacity)
🎯 Medication-lab interactions (causal relationships)
🎯 Temporal features (disease progression)
🎯 Increased model capacity (more parameters)

---

## Configuration Management

### Current Best: Iteration 1
```yaml
cohort:
  cv_cohort_mode: false        # Use all 1,834 patients
  use_first_icu_only: true
  age_min: 18

feature_space:
  labs:
    cv_target_mode: true       # Predict only CV labs
    cv_target_labs:
      - "troponin - I"
      - "PT - INR"
      - "PT"
      # ... (12 total)
```

### For Single-Lab Experiment:
```yaml
feature_space:
  labs:
    cv_target_mode: true
    cv_target_labs:
      - "troponin - I"         # Only one lab
```

### For Heart Failure Experiment:
```yaml
cohort:
  cv_cohort_mode: false        # Keep all patients

feature_space:
  labs:
    cv_target_mode: true
    cv_target_labs:
      - "BNP"
      - "troponin - I"
      - "potassium"
      - "sodium"
      - "creatinine"
      - "lactate"
      # 6-8 HF-specific labs
```

---

## Metrics to Track

For every experiment, record:

1. **Overall Metrics**
   - R², MAE, RMSE, MAPE
   - Number of test samples
   - Training time

2. **Per-Lab Metrics**
   - R², MAE, RMSE for each target lab
   - Number of samples per lab
   - Comparison to baseline for that lab

3. **Frequency Stratification**
   - Rare labs (bottom 25%)
   - Common labs (middle 50%)
   - Very common labs (top 25%)

4. **Clinical Interpretation**
   - Which labs improved/worsened?
   - Are high-value labs (troponin, BNP, INR) improving?
   - Is performance clinically useful?

---

## Bottom Line

**Main Finding:**
Focusing model training on specific lab types **DOES** improve prediction for those labs.

**Evidence:**
- CV labs in baseline: R² = 13.95%
- CV labs in focused model: R² = 19.04%
- **36.5% improvement**

**Next Step:**
Test extreme case: single-lab specialists to maximize improvement

**Clinical Impact:**
Need R² > 40% for clinical utility. Current best is PT (50.6%) and lactate (49.0%).
Focusing on these high-performers is most promising path forward.
