# Cereal 8 — Future Work

Oct 26, 2026 · Anas Alsirafy

Fix the specialist bug, find out whether specialists can beat the monolithic 0.468, then clean up evaluation so the maize numbers can be published as the first benchmark on this dataset. The order is a suggestion; Prof. Habib sets the final one.

## High priority

- [ ] Fix `train_specialist()` so each specialist keeps its best phase. Re-run fold 1 and compare against the monolithic fold-1 score.
- [ ] If specialists still lose, retrain S1–S4 one-vs-rest on all 2,355 images so they see multi-disease leaves.
- [ ] Build the side-by-side ablation table: per-label F1, macro-F1, micro-F1, Hamming loss and AUC for monolithic vs specialist.

## Medium priority

- [ ] Run the duplicate screen on maize and switch to grouped multilabel folds.
- [ ] Move threshold tuning to an inner validation split.
- [ ] Repeat the best configuration with 3–5 seeds and report mean ± s.d.
- [ ] Record exact per-label precision, recall and F1 for the monolithic model.

## Low priority

- [ ] Write up a short paper: first deep-learning benchmark on the Wiesner-Hanks 8-label field set, plus the monolithic-vs-specialist ablation.
- [ ] Try a smaller backbone (EfficientNet-B0, MobileNetV3) for edge deployment, as was done for cassava.
- [ ] Add maize to the cross-dataset protocol so it can be compared with the five Paper 2 datasets on equal terms.
