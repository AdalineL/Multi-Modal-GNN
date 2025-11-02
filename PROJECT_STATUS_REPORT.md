# Project Status Report: EHR Graph-Based Lab Test Recommendation

**Course**: Graph Theory Project
**Date**: November 2, 2025
**Track**: Graph Neural Networks (GNN)

---

## 1. Objective / Hypothesis / Prediction Task (2 pts)

### Objective Statement
**Create a multi-modal heterogeneous graph to predict links (edges) between patient nodes and lab result nodes**, specifically determining which laboratory tests should be ordered for ICU patients based on their clinical context (diagnoses and medications).

### Task Type
**Graph Link Prediction** (Graph-Based Edge Prediction)

**Important Distinction**: This is NOT a traditional classification task. It is a **graph-based link prediction task** where:
- The graph structure itself is the model input
- We predict the existence of edges between node pairs (patient ↔ lab)
- Predictions leverage graph topology and message passing, not just node features

<!-- ### Hypothesis
Incorporating **medication dosage information as edge weights** in the heterogeneous graph will improve the model's ability to predict patient-lab edges by providing richer clinical context about treatment intensity compared to binary edge presence alone. -->

### Target Outcome
**Predicted Links**: For each (patient, lab) pair, predict whether an edge should exist
- **Output**: Probability [0, 1] that a patient should receive a specific lab test
- **Graph-Based**: Predictions use 2-hop message passing through diagnoses and medications
- **NOT predicting**: Lab values, thresholds, or abnormality — only edge existence

### Population/Scope
- **Population**: Adult ICU patients from the eICU Collaborative Research Database
- **Cohort Size**: 1,834 patients across 200+ hospitals
- **Clinical Context**: Multi-center ICU setting with diverse patient conditions and lab ordering patterns
- **Graph Scope**: Multi-modal with 3 clinical node types (patient, lab, diagnosis) + medication context

---

## 2. Materials (8 pts)

### 2.1 Data & Unit of Analysis (2 pts)

#### Data Source
- **Dataset**: eICU Collaborative Research Database (demo version 2.0.1)
- **Source**: PhysioNet (https://physionet.org/content/eicu-crd-demo/2.0.1/)
- **Time Window**: ICU admission period (snapshot within stay, not longitudinal)
- **Ethics**: De-identified, publicly available research dataset

#### Cohort Definition
**Inclusion Criteria**:
- Adult ICU patients (age ≥ 18)
- At least 1 lab measurement recorded
- Complete demographic data available

**Exclusion Criteria**:
- Pediatric patients (age < 18)
- ICU stays with no lab measurements
- Missing critical identifiers

#### Unit of Analysis
**Primary Unit**: Patient-Lab Edge (Graph Link)
- Each observation represents a **potential link** in the heterogeneous graph between a patient node and a lab node
- **NOT traditional observations**: These are graph edges, evaluated in context of the full graph structure
- **Graph Context**: Predictions consider 2-hop neighborhoods (patient → diagnosis → other patients with similar conditions)
- **Total Positive Edges**: 61,484 existing patient-lab connections
- **Negative Sampling**: Equal number of non-existing edges sampled for balanced link prediction (NOT classification — this is graph-specific)

#### Key Counts

| Entity | Count | Description |
|--------|-------|-------------|
| **Patients** | 1,834 | Unique ICU patient stays |
| **Lab Types** | 50 | Unique laboratory tests (glucose, sodium, hemoglobin, etc.) |
| **Diagnoses** | 114 | Unique ICD-9 diagnosis codes (3-digit collapsed) |
| **Medications** | 100 | Unique drug names (normalized, top-100 most common) |
| **Patient-Lab Edges** | 61,484 | Existing patient-lab test relationships |
| **Diagnosis Edges** | 5,421 | Patient-diagnosis associations |
| **Medication Edges** | 15,933 | Patient-medication prescriptions |
| **Total Edges** | 165,676 | All edges (including bidirectional) |

#### Missingness Overview
- **Lab Values**: 0% missing in existing edges (only recorded measurements included)
- **Medication Dosages**: 8.9% missing (14,522/15,933 have dosage information)
  - Missing dosages imputed with neutral value (0.5)
- **Demographics**: <1% missing (excluded patients with missing age/gender)
- **Graph Structure**: Intentional sparsity (patient-lab density = 0.67)

---

### 2.2 Network Inputs (2 pts)

#### Node Types and Definitions

**Multi-Modal Heterogeneous Graph** (3 primary clinical node types + medication context):

| Node Type | Count | Representation | Role in Graph |
|-----------|-------|----------------|---------------|
| **Patient** | 1,834 | Integer ID (0-1833) | Source nodes for link prediction |
| **Lab Result** | 50 | Integer ID (0-49) | Target nodes for link prediction (e.g., "Glucose", "Sodium", "Hemoglobin") |
| **Diagnosis** | 114 | Integer ID (0-113) | Context nodes — patient clinical conditions (ICD-9 codes, 3-digit) |
| **Medication** | 100 | Integer ID (0-99) | Context nodes — treatment patterns (normalized drug names) |

**Prediction Focus**: Links between **Patient ↔ Lab** nodes (primary task)
- Diagnoses and medications provide clinical context through message passing
- **NOT predicting**: Patient-diagnosis or patient-medication edges (these are given)

**Node Features (Graph-Specific)**:
- **All nodes use learnable embeddings** (128-dim lookup tables, NOT handcrafted features)
- Embeddings are **parameters of the GNN**, optimized via backpropagation through message passing
- Embedding dimension: 128 for all node types
- Initialization: Xavier uniform
- **Patient nodes**: Additional 3-layer MLP transformation + L2 normalization (clusters similar patients)

#### Edge Types and Connections

**Heterogeneous Graph Structure** (6 edge types, bidirectional):

| Edge Type | Count | Direction | Weighted | Description |
|-----------|-------|-----------|----------|-------------|
| `(patient, has_lab, lab)` | 61,484 | Forward | **No** | Patient received this lab test (link prediction target) |
| `(lab, has_lab_rev, patient)` | 61,484 | Reverse | **No** | Lab test ordered for patient (reverse for message passing) |
| `(patient, has_diagnosis, diagnosis)` | 5,421 | Forward | No | Patient has this diagnosis |
| `(diagnosis, has_diagnosis_rev, patient)` | 5,421 | Reverse | No | Diagnosis applies to patient |
| `(patient, has_medication, medication)` | 15,933 | Forward | **Yes (dosage)** | Patient prescribed this medication |
| `(medication, has_medication_rev, patient)` | 15,933 | Reverse | **Yes (dosage)** | Medication prescribed to patient |

**Edge Attributes**:
1. **Patient-Lab Edges**:
   - **Unweighted** — no edge attributes (we're predicting edge existence, not lab values)
   - Lab values are NOT used as features since they're unknown until the edge exists
   - Link prediction focuses on "should this test be ordered?" not "what will the value be?"

2. **Patient-Medication Edges**:
   - Normalized medication dosage [0, 1] per drug (edge attribute)
   - Extraction: "5 mg" → 5.0 (numeric)
   - Normalization: Min-max per medication
   - Coverage: 91.1% (14,522/15,933 edges)
   - Missing: Imputed with 0.5 (neutral)

**Directed/Undirected**:
- Structurally undirected (bidirectional edges for message passing)
- Semantically directed (patient → lab represents "patient needs this lab")

**Temporal Windows**:
- Snapshot approach (single ICU stay aggregation)
- No temporal dynamics within stay
- All events during ICU stay collapsed to static graph

#### Network Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Nodes** (\|V\|) | 2,098 |
| **Total Edges** (\|E\|) | 165,676 (including reverse) |
| **Patient-Lab Density** | 0.6705 (61,484 / 91,700 possible) |
| **Avg Patient Degree** (has_lab) | 33.52 ± 9.56 (median: 35) |
| **Avg Patient Degree** (has_diagnosis) | 2.96 ± 2.84 (median: 2) |
| **Avg Patient Degree** (has_medication) | 8.69 ± 8.02 (median: 8) |
| **Graph Diameter** | Not computed (heterogeneous, multi-modal) |
| **Clustering Coefficient** | N/A (heterogeneous graph, different semantics) |

---

### 2.3 Track-Specific Inputs: GNN Track (2 pts)

#### Graph Tensors

**Adjacency Representation**: Edge list format (COO)
```python
# Patient-Lab edges (primary prediction target)
edge_index_patient_lab = torch.Tensor([
    [patient_indices],  # Source nodes: [61,484]
    [lab_indices]       # Target nodes: [61,484]
], dtype=torch.long)  # Shape: [2, 61,484]

# Patient-Diagnosis edges
edge_index_patient_dx = torch.Tensor([2, 5,421])

# Patient-Medication edges
edge_index_patient_med = torch.Tensor([2, 15,933])
```

**Node Feature Matrices**: None (all learnable embeddings)
```python
# Patient embeddings (initialized during model creation)
patient_embeddings = nn.Embedding(
    num_embeddings=1834,
    embedding_dim=128
)  # Parameters: 234,752

# Similarly for lab, diagnosis, medication nodes
# Total embedding parameters: 268,544
```

**Edge Feature Matrices**:
```python
# Patient-Lab edges: NO edge attributes (unweighted, link prediction target)
# We're predicting edge existence, not lab values

# Patient-Medication edge features
medication_dosages = torch.Tensor([15,933, 1])  # [0, 1] normalized dosages
```

#### Categorical/Temporal Feature Encoding

**Categorical Features**:
- **Not used** — all nodes represented by learnable integer IDs
- Original categorical data (gender, ethnicity, drug names) preprocessed but discarded
- Rationale: Learnable embeddings outperformed one-hot encoding (prevented overfitting)

**Temporal Features**:
- **Snapshot approach**: All events within ICU stay aggregated
- Time offsets (e.g., lab measurement time, drug start time) discarded
- Future work: Temporal GNN with RNN/Transformer layers

**Numerical Features**:
- Lab values: **NOT used as edge attributes** (only used to identify existing edges in preprocessing)
  - Lab values were z-score normalized during preprocessing but discarded for graph construction
  - Rationale: Link prediction predicts edge existence, not values
- Medication dosages: Min-max normalization per drug (used as edge attributes)
  - `normalized = (value - min) / (max - min)`
  - Ensures [0, 1] range, comparable across medications
  - Used as edge weights to provide treatment intensity context

#### Heterogeneous Graph Structure

**Node Types**: 4 (`patient`, `lab`, `diagnosis`, `medication`)

**Edge Types**: 6 (3 forward + 3 reverse)

**Metadata Tensor**:
```python
metadata = (
    ['patient', 'lab', 'diagnosis', 'medication'],  # Node types
    [
        ('patient', 'has_lab', 'lab'),
        ('lab', 'has_lab_rev', 'patient'),
        ('patient', 'has_diagnosis', 'diagnosis'),
        ('diagnosis', 'has_diagnosis_rev', 'patient'),
        ('patient', 'has_medication', 'medication'),
        ('medication', 'has_medication_rev', 'patient')
    ]  # Edge types
)
```

---

### 2.4 Variables / Labels (2 pts)

**Note**: Traditional "independent/dependent variable" terminology from statistics doesn't directly map to graph link prediction. Below we clarify the graph-specific equivalents.

#### Graph Links (What We're Predicting)

**Definition**: Each prediction target is a **potential edge** between a (patient, lab) node pair
- **Positive links**: Existing edges in graph (patient actually received this lab test)
- **Negative links**: Sampled non-existing edges (patient did NOT receive this lab test)
- **Graph context**: Each link evaluated using full graph structure via message passing

**Total Links** (per split):
- Train: 86,076 edges (43,038 positive + 43,038 negative)
- Validation: 18,444 edges (9,222 positive + 9,222 negative)
- Test: 18,448 edges (9,224 positive + 9,224 negative)

#### Graph-Based "Features" (What the Model Uses)

**NOT traditional features** — GNNs use graph structure and learned embeddings:

**Node Embeddings** (learned during training):
- Patient embedding (128-dim, updated via backprop)
- Lab embedding (128-dim, updated via backprop)
- Diagnosis embeddings (114 nodes × 128-dim)
- Medication embeddings (100 nodes × 128-dim)

**Graph Structure** (message passing input):
- 1-hop neighbors: Patient's existing labs, diagnoses, medications
- 2-hop neighbors: Similar patients (via shared diagnoses), co-occurring labs

**Edge Attributes**:
- Patient-Lab edges: **None** (unweighted, predicting edge existence)
- Medication dosages: Min-max normalized [0,1] (continuous, provides treatment context)

**Key Difference from Traditional ML**:
- Traditional ML: Features are fixed inputs (age, gender, etc.)
- GNN: Embeddings are LEARNED parameters + graph topology provides structure

#### Link Labels (Prediction Target)

**Task**: Graph link prediction (NOT traditional binary classification)
- **Label**: Edge existence (0 or 1)
  - 1 = Link exists (patient-lab edge present in graph)
  - 0 = Link does not exist (no edge between this patient-lab pair)
- **Output**: Edge probability [0, 1] via sigmoid activation
- **Evaluation context**: Graph-aware (considers structural properties)

**Important**: While we output binary labels, this is evaluated in the context of the FULL GRAPH, not as independent samples. Graph structure matters.

#### Negative Sampling (Graph-Specific)

**Why Needed**: Link prediction requires both positive (existing edges) and negative (non-existing edges) examples. Unlike traditional classification, we must explicitly sample negatives from the large space of possible edges.

**Method**: Random non-existing edge sampling
```python
# For each positive edge (patient_i, lab_j) in split:
#   1. Sample random patient_k and lab_l
#   2. Check if (patient_k, lab_l) exists in FULL graph
#   3. If not, add as negative sample
#   4. Repeat until num_negatives == num_positives
```

**Key Properties**:
- Balanced: 50% positive, 50% negative edges
- **Graph-aware**: Excludes ALL existing edges (train + val + test) — critical to avoid false negatives
- **True negatives**: Negatives are genuinely non-existent relationships (not mislabeled positives)
- Seed-controlled: Reproducible (seed=42)

#### Link Distribution (NOT "Class Balance")

**Note**: We avoid "class" terminology since this is graph link prediction, not classification.

**Training Links**:
- Positive edges: 43,038 (50.0%)
- Negative edges: 43,038 (50.0%)
- **Balanced by design** via negative sampling

**Validation Links**:
- Positive edges: 9,222 (50.0%)
- Negative edges: 9,222 (50.0%)

**Test Links**:
- Positive edges: 9,224 (50.0%)
- Negative edges: 9,224 (50.0%)

**No Class Weighting**: Not applied (balanced by construction)

---

## 3. Methods (8 pts)

### 3.1 Graph Construction & Edge Strength (2 pts)

#### Edge Creation Rules

**Patient-Lab Edges**:
- **Rule**: Include edge if patient has ≥1 recorded measurement for that lab type
- **Threshold**: None (all recorded measurements included)
- **Weighting**: Z-score normalized lab value
  - Mean and std computed per lab type on training set
  - Applied to all splits

**Patient-Diagnosis Edges**:
- **Rule**: Include edge if diagnosis appears in patient's diagnosis table
- **ICD-9 Collapsing**: 3-digit codes (e.g., "428.0" → "428")
- **Frequency Filter**: Keep only top-114 most common diagnoses (min 5 patients)
- **Weighting**: Unweighted (binary presence)

**Patient-Medication Edges**:
- **Rule**: Include edge if medication prescribed during ICU stay
- **Drug Normalization**: Generic name extraction (e.g., "Aspirin 81mg" → "aspirin")
- **Frequency Filter**: Keep only top-100 most common medications (min 5 patients)
- **Weighting**: Normalized dosage [0, 1] per medication
  - **Extraction**: Parse numeric value from text (e.g., "5 mg" → 5.0)
  - **Normalization**: Min-max per drug
  - **Coverage**: 91.1% (14,522/15,933)
  - **Missing**: Imputed with 0.5 (neutral)

#### Temporal Aggregation
- **Approach**: Snapshot (all events within ICU stay)
- **Lab Values**: Mean of multiple measurements
- **Medications**: Most recent dosage
- **Diagnoses**: Union of all diagnoses

#### Handling Sparsity/Density

**Sparsity Management**:
- **Lab Selection**: Top-50 most common labs (coverage: 99.8% of patients)
- **Patient-Lab Density**: 0.67 (relatively dense due to common lab panels)
- **Diagnosis Density**: 0.032 (sparse, expected for rare conditions)
- **Medication Density**: 0.087 (moderately sparse)

**No additional filtering**:
- No k-NN edge pruning (graph already sparse enough)
- No backbone extraction (preserves all clinical information)

#### Bidirectional Edges
- All edge types created bidirectionally for GNN message passing
- Forward: `(patient, has_lab, lab)`
- Reverse: `(lab, has_lab_rev, patient)`
- Rationale: Allows information flow in both directions

---

### 3.2 Network Metrics & Their Use (1 pt)

#### Exploratory Metrics (Graph Structure Analysis)

**Node-Level Metrics**:
- **Patient Degree Distribution**:
  - has_lab: Mean=33.52, Median=35 (most patients have full lab panels)
  - has_diagnosis: Mean=2.96, Median=2 (sparse diagnoses)
  - has_medication: Mean=8.69, Median=8 (moderate polypharmacy)
  - **Use**: Informed degree-aware hybrid architecture (threshold=6)

**Graph-Level Metrics**:
- **Density**: 0.67 (patient-lab subgraph)
  - **Use**: Confirms graph is not overly sparse, suitable for GNN
- **Connected Components**: 1 (fully connected)
  - **Use**: Ensures all nodes reachable via message passing

**Why These Matter**:
- Degree distribution revealed bimodal pattern → motivated degree-aware prediction
- High density validated GNN applicability (sparse graphs struggle with message passing)

#### Metrics NOT Used as Features
- Centrality (betweenness, eigenvector) — not computed
- Community detection — not relevant for heterogeneous clinical graph
- Path lengths — not used (2-layer GNN captures 2-hop neighborhood)

**Rationale**: These metrics are descriptive, not predictive for link prediction task.

---

### 3.3 Modeling Approach: GNN Track (3 pts)

#### Architecture

**Model Family**: Heterogeneous Relational Graph Convolutional Network (Hetero-RGCN)

**Depth**: 2 GNN layers + 3-layer patient MLP

**Message Passing Scope**:
- **Layer 1**: 1-hop neighborhood aggregation
  - Patient → {labs, diagnoses, medications}
  - Lab → {patients}
  - Diagnosis → {patients}
  - Medication → {patients}

- **Layer 2**: 2-hop neighborhood aggregation
  - Patient → {similar patients via shared labs/diagnoses/medications}
  - Lab → {co-occurring labs via shared patients}

**Aggregation Function**: Mean pooling (SAGEConv)
- `message = MEAN([neighbor_embeddings])`
- Scale-invariant (handles variable-degree nodes)

**Architecture Diagram**:
```
Input: Patient ID, Lab ID
  ↓
Node Embeddings (Learnable)
├─ Patient: nn.Embedding(1834, 128)
├─ Lab: nn.Embedding(50, 128)
├─ Diagnosis: nn.Embedding(114, 128)
└─ Medication: nn.Embedding(100, 128)
  ↓
Patient MLP Transform (3 layers)
├─ Linear(128→128) + BatchNorm + ReLU + Dropout
├─ Linear(128→128) + BatchNorm + ReLU + Dropout
└─ Linear(128→128) + L2 Norm
  ↓
GNN Layer 1 (HeteroConv)
├─ SAGEConv for each edge type
└─ Mean aggregation + BatchNorm + ReLU + Dropout
  ↓
GNN Layer 2 (HeteroConv)
├─ SAGEConv for each edge type
└─ Mean aggregation + BatchNorm + ReLU
  ↓
Concatenate [patient_emb; lab_emb]  (256-dim)
  ↓
Link Prediction MLP
├─ Linear(256→64) + ReLU + Dropout
├─ Linear(64→32) + ReLU + Dropout
└─ Linear(32→1) + Sigmoid
  ↓
Output: Edge probability [0, 1]
```

**Total Parameters**: 483,970
- Embeddings: 268,544 (55.5%)
- Patient MLP: 82,048 (17.0%)
- R-GCN layers: 114,818 (23.7%)
- Prediction head: 18,560 (3.8%)

#### Readout

**Node-Level Embeddings**:
- After 2 GNN layers, extract final embeddings for patient and lab nodes

**Edge Prediction**:
```python
# Concatenate patient and lab embeddings
edge_repr = concat([patient_emb, lab_emb])  # 256-dim

# MLP prediction
prob = MLP(edge_repr)  # → [0, 1]
```

**No Graph-Level Readout**: Task is link prediction, not graph classification

#### Loss Function

**Primary Loss**: Binary Cross-Entropy (BCE)
```python
loss = -[y * log(p) + (1-y) * log(1-p)]
```
where:
- `y` = ground truth label (0 or 1)
- `p` = predicted probability (sigmoid output)

**Lab-Wise Reweighting** (optional, currently used):
```python
# Compute variance per lab type on training set
lab_variances = {lab_id: var(train_values)}

# Inverse variance weights (normalized)
weights = 1 / (lab_variances + epsilon)
weights = weights * num_labs / sum(weights)

# Apply to loss
weighted_loss = loss * weights[lab_id]
```

**Rationale**: Prevents high-variance labs (e.g., glucose) from dominating loss

#### Multi-Task Setup
**None**: Single task (link prediction for patient-lab edges)

**Future Directions**:
- Multi-task: Joint prediction of labs + outcomes (mortality, readmission)
- Multi-label: Predict multiple lab panels simultaneously

---

### 3.4 Experimental Design: GNN Track (2 pts)

#### Train/Val/Test Split Strategy

**Split Type**: Edge-level random split (transductive setting)

**Split Ratios**:
- Train: 70% (43,038 edges)
- Validation: 15% (9,222 edges)
- Test: 15% (9,224 edges)

**Split Procedure**:
```python
# Random permutation of all 61,484 patient-lab edges
perm = torch.randperm(61,484, seed=42)

# Assign to splits
train_edges = perm[:43,038]
val_edges = perm[43,038:52,260]
test_edges = perm[52,260:]
```

**Transductive vs Inductive**:
- **Current**: Transductive (test edges in graph structure during training)
- **Consequence**: Model sees structural info but not labels of test edges
- **Justification**: Standard practice for GNN link prediction (GraphSAGE, GAT papers)

**Patient Overlap**:
- **Expected**: Same patient can have different lab edges in different splits
- **Example**: Patient 42's glucose → Train, sodium → Test
- **Valid**: Edge-level split, not patient-level holdout

#### Negative Sampling Strategy (Detailed)

**Training**:
- For each batch of positive edges, sample equal number of negatives
- Negatives sampled from non-existing edges in FULL graph
- New negatives each epoch (different seed)

**Validation/Test**:
- Negative edges sampled once (seed=42)
- Same negatives used across all validation/test runs
- Balanced: 50% positive, 50% negative

#### Hyperparameter Plan

**Fixed Hyperparameters**:
```yaml
hidden_dim: 128          # Embedding dimension
num_layers: 2            # GNN depth
dropout: 0.3             # Regularization
learning_rate: 0.001     # Adam optimizer
batch_size: 1024         # For negative sampling
epochs: 100              # With early stopping
early_stopping: 15       # Patience
```

**Tuning Strategy**:
- **Current**: Manual tuning based on validation loss
- **Future**: Grid search over:
  - `hidden_dim: [64, 128, 256]`
  - `num_layers: [1, 2, 3]`
  - `dropout: [0.1, 0.3, 0.5]`
  - `learning_rate: [0.0001, 0.001, 0.01]`

**Seed Control**:
- Random seed: 42 (PyTorch, NumPy, Python random)
- Ensures reproducibility across runs

#### Ablations/Baselines

**Baseline 1**: Model without medication dosage edge weights
- **Setup**: Same architecture, but dosage=None
- **Result**: AUROC=0.9238, Recall=78.15%

**Ablation 2**: Model WITH medication dosage edge weights (current)
- **Setup**: Dosage normalized [0,1] per drug
- **Result**: AUROC=0.9275, Recall=79.71% (+1.98% improvement)

**Baseline 3** (Future): Simple heuristics
- **Co-occurrence**: Predict lab if other similar patients have it
- **Frequency**: Predict top-K most common labs
- **Expected**: GNN should outperform by large margin

**Baseline 4** (Future): Non-graph ML
- **Logistic Regression**: Patient demographics → lab prediction
- **Random Forest**: Diagnoses + medications → lab prediction
- **Expected**: GNN leverages graph structure better

---

## 4. Results (Preliminary/Partial) (1 pt)

### 4.1 Primary Performance Metrics

**Link Prediction Performance** (Test Set, N=18,448):

| Metric | Dosage-Weighted<br> |
|--------|-----------------------------|
| **AUROC** | **0.9275** |
| **AUPRC** | **0.9332** |
| **Accuracy** | **84.47%** |
| **Precision** | 88.10% |
| **Recall** | **79.71%** |
| **F1 Score** | **83.69%** |

**Takeaway**: Adding medication dosage as edge weights improves recall (+1.98%) and AUROC (+0.40%), with minimal impact on other metrics. The model better identifies patient-lab relationships when treatment intensity is encoded.

---

### 4.2 Confusion Matrix Analysis

**Test Set Confusion Matrices**:

| Model | TP | FP | TN | FN | Sensitivity | Specificity |
|-------|----|----|----|----|-------------|-------------|
| **Baseline** | 7,209 | 924 | 8,300 | 2,015 | 78.2% | 90.0% |
| **Dosage** | **7,352** | 993 | 8,231 | **1,872** | **79.7%** | 89.2% |
| **Δ** | **+143** | +69 | -69 | **-143** | **+1.5%** | -0.8% |

**Takeaway**: Dosage model identifies **143 additional true positives** (better detection) at the cost of 69 additional false positives. The net effect is positive (+1.98% recall).

---

### 4.3 Ranking Performance (Recall@K)

**Recall@K Metrics** (What % of true positives are in top-K predictions):

| K | Recall@K |
|---|----------|
| 10 | 0.11% |
| 20 | 0.22% |
| 50 | 0.54% |
| 100 | 1.08% |

**Interpretation**:
- Low absolute values expected (9,224 total positives / large search space)
- Stable across baseline and dosage models
- Indicates model ranks true positives relatively evenly (not concentrated at top)

---

### 4.4 Training Convergence

**Training Curves** (100 epochs, early stopping at epoch 100):

```
Best Validation Loss: 0.3411 (epoch 100)
Final Test Loss: 0.3487

Training Time: 9 minutes 28 seconds (CPU)
Convergence: Smooth, no overfitting detected
```

**Early Stopping**: Did not trigger (validation loss continued improving)

---

### 4.5 Dosage Coverage Analysis

**Medication Dosage Extraction**:

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Medication Edges** | 15,933 | 100% |
| **With Dosage Info** | 14,522 | **91.1%** ✓ |
| **Missing Dosage** | 1,411 | 8.9% |

**Dosage Distribution** (example: Aspirin):
- Min: 81 mg
- Max: 325 mg
- Normalized: [0, 1] via min-max scaling

**Takeaway**: High coverage (91.1%) validates dosage as reliable edge attribute.

---

### 4.6 Network Statistics Summary

**Graph Structure**:

| Metric | Value |
|--------|-------|
| **Nodes** | 2,098 (1,834 patients + 264 clinical entities) |
| **Edges** | 165,676 (including bidirectional) |
| **Patient-Lab Density** | 0.6705 |
| **Avg Patient Degree** | 45.17 (across all edge types) |

**Figure 1**: Patient degree distribution (has_lab edges)
```
     Count
50  |  ████████ (n=412)
40  |  ████████████ (n=523)
30  |  ██████████ (n=445)
20  |  ████ (n=287)
10  |  ██ (n=127)
 0  |  █ (n=40)
```
**Takeaway**: Most patients have 30-40 lab tests (comprehensive panels), validating GNN applicability.
<!-- 
---

### 4.7 Model Comparison Summary

**Dosage Impact**:
```
AUROC: 0.9238 → 0.9275 (+0.40%)
Recall: 78.15% → 79.71% (+1.98%)
True Positives: 7,209 → 7,352 (+143)
```

**Statistical Significance**: Not yet tested (future: McNemar's test, DeLong test)

**Clinical Significance**: 143 additional correct predictions could translate to better lab ordering recommendations in production.

--- -->

## 5. Problems Encountered & Potential Solutions (1 pt)

### 5.1 Data-Related Problems

#### Problem 1: Missing Medication Dosages (8.9%)
**Impact**: Some medication edges lack dosage information, requiring imputation.

**Concrete Solution**:
- ✅ **Current**: Impute with neutral value (0.5)
- **Alternative 1**: Learn dosage imputation as auxiliary task
- **Alternative 2**: Use dosage presence as binary flag (has_dosage: 0/1)
- **Alternative 3**: Exclude edges without dosage (reduces data by 8.9%)

**Status**: Currently using neutral imputation (0.5), minimal performance impact.

---

#### Problem 2: Imbalanced Lab Frequency
**Impact**: Some labs appear in 99% of patients (glucose), others in <5% (troponin).

**Concrete Solution**:
- ✅ **Current**: Lab-wise loss reweighting (inverse variance)
- **Alternative**: Focal loss (focuses on hard examples)
- **Alternative**: Stratified sampling by lab frequency

**Status**: Lab-wise reweighting implemented and effective.

---

### 5.2 Modeling Problems

#### Problem 3: Transductive Setting (Data Leakage Risk)
**Impact**: Test edges remain in graph structure during training (structural info leaks).

**Concrete Solution**:
- **Current**: Transductive (standard for GNN link prediction papers)
- **Alternative 1**: Inductive setting — remove test edges from training graph
  ```python
  train_graph = create_subgraph(data, edge_mask=train_mask | val_mask)
  test_graph = full_graph  # Evaluate on full graph
  ```
- **Alternative 2**: Patient-level holdout — split patients, not edges
  ```python
  train_patients = patients[:1300]
  test_patients = patients[1300:]
  # All edges for test patients hidden during training
  ```

**Status**: Transductive setting acceptable for current experiments; inductive version planned for future work.

---

#### Problem 4: Low Recall@K Values
**Impact**: Top-K predictions don't concentrate true positives well.

**Concrete Solution**:
- **Root Cause Analysis**: Large search space (1,834 × 50 = 91,700 possible edges)
- **Alternative 1**: Rerank predictions using clinical rules (e.g., prioritize glucose for diabetics)
- **Alternative 2**: Multi-task learning (jointly predict labs + diagnoses)
- **Alternative 3**: Attention mechanism to weight neighbors differently

**Status**: Acceptable for balanced classification; investigate ranking improvements.

---

### 5.3 Computational Problems

#### Problem 5: CPU-Only Training (Slow)
**Impact**: Training takes ~10 minutes per run, limiting hyperparameter search.

**Concrete Solution**:
- **Current**: CPU training on MacBook (M-series chip)
- ✅ **Solution Implemented**: Efficient batch processing, early stopping
- **Alternative 1**: GPU acceleration (CUDA)
  - Requires: GPU-enabled machine or cloud instance
  - Expected speedup: 5-10×
- **Alternative 2**: Mixed-precision training (FP16)
  - Reduces memory, speeds up training
  - PyTorch AMP support

**Status**: CPU acceptable for demo dataset; GPU recommended for full eICU (200K patients).

---

### 5.4 Experimental Design Problems

#### Problem 6: No Confidence Intervals
**Impact**: Cannot assess statistical significance of improvements.

**Concrete Solution**:
- **Current**: Single run with seed=42
- **Alternative**: 10-fold cross-validation or bootstrap resampling
  ```python
  for seed in [1, 2, 3, 4, 5]:
      model = train(seed=seed)
      results[seed] = evaluate(model)

  auroc_mean = mean(results['auroc'])
  auroc_95ci = [quantile(0.025), quantile(0.975)]
  ```
- **Alternative**: McNemar's test for confusion matrix differences

**Status**: Planned for final report.

---

### 5.5 Future Directions

#### Next Steps (Concrete Action Items)

1. **Inductive Evaluation** (Priority: High)
   - Implement patient-level holdout
   - Compare transductive vs inductive AUROC
   - **Timeline**: Next 1 week

2. **Temporal Modeling** (Priority: Medium)
   - Add time-series component (sequential lab orders)
   - Use RNN/Transformer layers
   - **Timeline**: 2-3 weeks

3. **Interpretability** (Priority: High)
   - GNNExplainer for edge predictions
   - Attention weight visualization
   - **Timeline**: 1 week

4. **Hyperparameter Tuning** (Priority: Medium)
   - Grid search over architecture parameters
   - Use Optuna or Ray Tune
   - **Timeline**: 2 weeks

5. **Full Dataset** (Priority: Low)
   - Scale to full eICU (200K patients)
   - Requires GPU cluster
   - **Timeline**: 4+ weeks

---

## Submission Checklist ✓

- ✅ Objective/hypothesis stated in one sentence with task type
- ✅ Data summary with unit of analysis, counts, and missingness
- ✅ Node/edge definitions + example features; adjacency/feature tensors (GNN)
- ✅ Variables/labels clearly defined and time-aligned (no leakage)
- ✅ Graph construction rules and edge weighting/thresholds
- ✅ Chosen metrics and how they're used
- ✅ Model spec (GNN), loss/diagnostics, confounders
- ✅ Experimental design: splits, baselines/ablations
- ✅ Preliminary results figure/table with one-line interpretation
- ✅ Problems + concrete next steps

---

## Appendix: Key Files

**Code**:
- `src/preprocess.py` — Data preprocessing, dosage extraction
- `src/graph_build.py` — Graph construction, edge weighting
- `src/model.py` — Hetero-RGCN architecture
- `src/train.py` — Training loop, negative sampling
- `src/evaluate.py` — AUROC, AUPRC, Recall@K metrics
- `src/compare_results.py` — Baseline vs dosage comparison

**Results**:
- `outputs/evaluation_results.json` — Test set metrics
- `outputs/comparison_report.txt` — Detailed comparison
- `outputs/training_history.json` — Training curves

**Documentation**:
- `README.md` — Full project documentation
- `conf/config.yaml` — Hyperparameters and settings

---

**End of Report**
