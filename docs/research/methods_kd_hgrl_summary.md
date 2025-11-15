# Methods Used in KD-HGRL (L-CoGNN)

Updated: 2025-10-18

This document summarizes the core methods and losses implemented in the KD-HGRL codebase for heterogeneous graph representation learning and hierarchical knowledge distillation.

- Repository: L-CoGNN
- Key code paths are referenced inline for quick lookup

---

## 1) HeCo-based Teacher Model (MyHeCo)

Files: `code/models/kd_heco.py`, `code/models/sc_encoder.py`, `code/models/contrast.py`

The teacher follows a HeCo-style dual-view architecture:

- Meta-path view encoder (`myMp_encoder`)
  - Per meta-path GCN layer stack + attention aggregation
  - File/symbols: `kd_heco.py:GCN`, `myMp_encoder`; GCN handles dense/sparse adjacencies robustly
- Schema view encoder (`mySc_encoder`)
  - Intra-type attention (neighbor sampling + attention)
  - Inter-type attention (attention over types)
  - File/symbols: `sc_encoder.py:intra_att`, `inter_att`, `mySc_encoder`
- Contrastive head (`Contrast`)
  - 2-layer projection head per view
  - Temperature-scaled similarity and symmetric contrastive loss
  - File/symbols: `contrast.py:Contrast`

Notation: Let z_mp = meta-path representation, z_sc = schema representation, and P be the positive adjacency mask (sparse COO tensor) derived from meta-path/pos. The model computes projections z′ = proj(z), similarities s = exp(sim(z′_mp, z′_sc)/τ), then normalizes row-wise, and applies a log-likelihood over positives. The final teacher loss is a convex combination of mp→sc and sc→mp terms with weight λ.

Key equations:
- Projection: z′ = MLP(z)
- Similarity (cosine scaled): s_ij = exp( (z′_i · z′_j) / (||z′_i||·||z′_j||·τ) )
- Loss (simplified): L = λ·L_mp→sc + (1−λ)·L_sc→mp, with each direction using row-normalized scores and a positive mask from P.

Implementation anchors:
- `kd_heco.py:MyHeCo.forward`
- `contrast.py:Contrast.forward`
- `sc_encoder.py:mySc_encoder.forward`

---

## 2) Heterogeneous Augmentation Expert (AugmentationTeacher)

File: `code/models/kd_heco.py` (class `AugmentationTeacher`)
Auxiliary: `code/training/hetero_augmentations.py`

Purpose: Train an expert on augmented graphs to provide augmentation-aware guidance (robustness signals) for the student.

Core components:
- Augmentation pipeline: `HeteroAugmentationPipeline` applies structure-aware meta-path connections, feature masking, etc., returning `(aug_feats, aug_info)`.
- Dual-view processing of both original and augmented inputs.
- Loss terms:
  - Contrastive loss on original graph: L_contrast(orig)
  - Contrastive loss on augmented graph: L_contrast(aug)
  - Invariance/divergence encouragement via cosine similarity between original and augmented embeddings:
    - L_div_mp = −α·cos(z_mp^orig, z_mp^aug)
    - L_div_sc = −α·cos(z_sc^orig, z_sc^aug)
  - Total: L_expert = 0.5·(L_contrast(orig)+L_contrast(aug)) + L_div_mp + L_div_sc

Guidance heads (used to guide student, not to prune):
- `mp_augmentation_guide`: predicts per-meta-path importance weights [P]
- `sc_augmentation_guide`: predicts per-schema-type importance weights [nei_num]
- `attention_importance`: node-level attention importance
- `cross_aug_learning`: structural predictors and `MultiheadAttention` to allocate attention from combined views
- Method `get_augmentation_guidance` returns a dict with `mp_importance`, `sc_importance`, `attention_importance`, `structure_importance`, `attention_weights`, and raw `augmentation_info`.

Implementation anchors:
- `kd_heco.py:AugmentationTeacher.forward`
- `kd_heco.py:AugmentationTeacher._generate_augmentation_guidance`
- `training/hetero_augmentations.py`

---

## 3) Compressed Student with Guidance (StudentMyHeCo)

File: `code/models/kd_heco.py` (class `StudentMyHeCo`)

Design:
- Compression: `student_dim = int(hidden_dim * compression_ratio)` (default 0.5)
- Same dual-view encoders at reduced width
- Optional integration of augmentation teacher guidance via learnable gates and fusion weights

Losses within student forward:
- Student contrastive loss L_student_contrast (same form as teacher, but at student_dim) controlled by flag `use_student_contrast_loss`.

Guidance integration:
- Meta-path and schema streams each have a gate:
  - Gate inputs: concat(student_output, teacher_guidance_projected) → sigmoid → per-dim gate
  - Fusion: (1−σ(w))·student + σ(w)·teacher_guidance·gate, with learnable scalar fusion weights per stream (`mp_fusion_weight`, `sc_fusion_weight`)
- Teacher→student projection layers (`mp_teacher_to_student`, `sc_teacher_to_student`) align teacher hidden_dim to student_dim when needed
- Method `get_teacher_aligned_representations` projects student reps to teacher space for KD alignment

Implementation anchors:
- `kd_heco.py:StudentMyHeCo.forward`
- `kd_heco.py:StudentMyHeCo._init_guidance_integration`

Flags (from `kd_params.py`):
- `use_student_contrast_loss` (default True)
- `student_compression_ratio` (default 0.5)

---

## 4) Dual-Teacher Knowledge Distillation (KD)

File: `code/models/kd_heco.py` (class `DualTeacherKD`)
Used by: `code/training/train_student.py`

Purpose: Align student to the main teacher; augmentation teacher provides separate robustness guidance.

Methods:
- Knowledge alignment head: maps student_dim → teacher_dim via MLP + LayerNorm (`self.knowledge_alignment`)
- `calc_knowledge_distillation_loss(...)`:
  - Get (teacher_mp, teacher_sc) from teacher and (student_mp, student_sc) from student
  - Align student to teacher dim
  - Apply MSE on L2-normalized embeddings for both views; temperature T scales the loss: L_KD = (MSE(norm(s_mp^→T), norm(t_mp)) + MSE(norm(s_sc^→T), norm(t_sc)))·0.5·T^2

Additional KD utility:
- `KLDiverge(teacher_logits, student_logits, T)`: classic KL divergence KD on soft targets (available but not the default in `calc_knowledge_distillation_loss`)

Config (from `kd_params.get_distillation_config`):
- `kd_temperature` (default 2.5)
- `use_kd_loss` toggle
- `main_distill_weight` (weight when combining into total loss)

Implementation anchors:
- `kd_heco.py:DualTeacherKD.calc_knowledge_distillation_loss`
- `kd_heco.py:KLDiverge`
- `training/train_student.py:StudentTrainer.train_epoch`

---

## 5) Link Reconstruction for Structure Modeling

File: `code/models/kd_heco.py` (function `link_reconstruction_loss`)
Used by: `training/train_student.py` (optional, flag-controlled)

- Positive edges sampled from meta-path adjacencies: `sample_edges_from_metapaths(mps, K)`
- Negative edges sampled avoiding observed edges: `sample_negative_edges`
- Loss: logistic loss on dot-product similarities for pos/neg edges:
  - L_pos = −E[log σ( (e_u·e_v)/T )], L_neg = −E[log σ( −(e_u·e_v)/T )]
  - L_link = L_pos + L_neg

Config flags (from `kd_params.py`):
- `use_link_recon_loss` (default True), `link_recon_weight`, `link_sample_rate`

Implementation anchors:
- `kd_heco.py:link_reconstruction_loss`
- `training/train_student.py:train_epoch`

---

## 6) Heterogeneous Augmentation Methods

File: `code/training/hetero_augmentations.py`

- Structure-aware meta-path connections (strength-controlled)
- Node feature masking / perturbation (as configured)
- Multi-augmentation support: `get_multiple_augmentations`
- Optional reconstruction loss helpers (scaffolded)

Entry points:
- `HeteroAugmentationPipeline.forward(feats, mps)` → `(aug_feats, aug_info)`
- `AugmentationTeacher` consumes `aug_feats` and `aug_info` to learn and to generate guidance

Config (from `get_augmentation_config` in `kd_params.py`):
- `use_meta_path_connections` (default True)
- `connection_strength` (default 0.2)

---

## 7) Evaluation Methods

File: `code/utils/evaluate.py`; Scripts: `code/evaluation/*`

- Node classification: train a simple logistic regression on frozen embeddings; report Accuracy, Macro-F1, Micro-F1; best epoch selected by validation Macro-F1
  - `evaluate_node_classification(...)`
- Link prediction: AUC and AP using dot-product edge scores and generated negatives
  - `evaluate_link_prediction(...)`, `generate_negative_edges(...)`
- Comprehensive evaluation harness: loads checkpoints and compares teacher vs student
  - `code/evaluation/evaluate_kd.py`
  - `code/evaluation/comprehensive_evaluation.py`

---

## 8) Loss Composition in Student Training

File: `code/training/train_student.py`

Total loss (epoch-dependent, flag-controlled):
- L_total = L_student
  + w_main·L_KD_from_main_teacher
  + w_aug·L_alignment_from_aug_teacher
  + w_link·L_link_reconstruction

Where:
- `w_main` = `args.main_distill_weight` (default 0.5)
- `w_aug` = `args.augmentation_weight` (default 0.5)
- `w_link` = `args.link_recon_weight` (default 0.6)

Toggles (from `kd_params.py`):
- `use_student_contrast_loss` (default True)
- `use_kd_loss` (default True)
- `use_augmentation_alignment_loss` (default True)
- `use_link_recon_loss` (default True)

---

## 9) Practical Notes and Safeguards

- Sparse/dense safety: `GCN.forward` and sampling utilities handle sparse COO tensors and dense tensors robustly (dimension checks, coalesce, fallbacks)
- Device handling: loaders and evaluators move tensors to CUDA when available; indices handled for both list and tensor forms
- Reproducibility: seeds and deterministic flags set across trainers; accumulation steps used for effective batch sizing
- Checkpoint structure: saved with `model_state_dict`, optimizer state, args, and guidance flags/ratios for student

---

## 10) Quick Map from Method → Code Symbol

- Contrastive learning: `contrast.py:Contrast`
- Meta-path encoder: `kd_heco.py:myMp_encoder`
- Schema encoder: `sc_encoder.py:mySc_encoder`
- Teacher: `kd_heco.py:MyHeCo`
- Augmentation expert: `kd_heco.py:AugmentationTeacher`, `training/hetero_augmentations.py`
- Student (compressed): `kd_heco.py:StudentMyHeCo`
- Dual-teacher KD: `kd_heco.py:DualTeacherKD`
- KL divergence KD: `kd_heco.py:KLDiverge`
- Link reconstruction: `kd_heco.py:link_reconstruction_loss`
- Edge sampling: `kd_heco.py:sample_edges_from_metapaths`, `kd_heco.py:sample_negative_edges`
- Node classification eval: `utils/evaluate.py:evaluate_node_classification`
- Link prediction eval: `utils/evaluate.py:evaluate_link_prediction`

---

If you want this summary rendered in the project README or a separate docs page with figures/equations, we can extend it with diagrams and example configs per dataset.
