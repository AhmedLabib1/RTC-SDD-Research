# Frozen Pretrained Backbone Feature Benchmark (Dev)

## Purpose

Compare frozen pretrained speech encoders as feature extractors before training a downstream classifier. This run extracts feature vectors from the RTCFake packed **Dev** split; it does **not** train a classifier and does not report F1, EER, accuracy, or the project's official evaluation score.

## Dataset and setup

- Dataset: RTCFake packed private dataset
- Split: `dev`
- Dev records processed: **15,204**
- Labels: `spoof = 0`, `bonafide = 1`
- Encoder weights: frozen pretrained checkpoints
- Pooling: temporal mean + standard deviation concatenated
- Feature dimensions: 1,536 for the Base encoders; 2,048 for XLS-R 300M
- Geometry diagnostics: computed on a balanced sample of 1,000 examples per class (2,000 total) from the extracted Dev vectors
- Similarity metric: cosine for silhouette, centroid distance, and compactness
- Neighbour metric: leave-one-out kNN label agreement at k=5

## Results

| Model | Checkpoint revision | Feature dims | Silhouette (cosine) ↑ | Centroid cosine distance ↑* | kNN label agreement@5 ↑ | Runtime (min) |
|---|---|---:|---:|---:|---:|---:|
| `facebook/wav2vec2-base` | `0b5b8e868dd8` | 1,536 | 0.0102273 | 0.0007190 | 0.6208 (62.08%) | 193.77 |
| `facebook/data2vec-audio-base` | `1f5a6404ec3d` | 1,536 | 0.0094220 | 0.0003346 | 0.5296 (52.96%) | 6.46 |
| `facebook/hubert-base-ls960` | `dba3bb02fda4` | 1,536 | 0.0086454 | 0.0003948 | 0.5630 (56.30%) | 6.33 |
| `microsoft/wavlm-base-plus` | `4c66d4806a42` | 1,536 | 0.0064677 | 0.0002764 | 0.5366 (53.66%) | 10.58 |
| `facebook/wav2vec2-xls-r-300m` | `1a640f32ac3e` | 2,048 | 0.0028739 | 0.0000693 | 0.5403 (54.03%) | 16.26 |

`↑` indicates that a larger value is preferable for the named diagnostic.  
`*` Centroid cosine distance is descriptive, not a universal quality score; its preferred direction depends on the intended geometry and should not be used alone to rank encoders.

## Class compactness

Compactness is mean cosine similarity to each class centroid (higher means tighter within-class grouping).

| Model | Spoof compactness ↑ | Bonafide compactness ↑ |
|---|---:|---:|
| `facebook/wav2vec2-base` | 0.970511 | 0.971123 |
| `facebook/hubert-base-ls960` | 0.980930 | 0.980414 |
| `microsoft/wavlm-base-plus` | 0.982254 | 0.982011 |
| `facebook/data2vec-audio-base` | 0.987237 | 0.985289 |
| `facebook/wav2vec2-xls-r-300m` | 0.991254 | 0.990972 |

## Interpretation and limitations

1. **Wav2Vec2 Base leads these preliminary label-geometry diagnostics**: it has the highest cosine silhouette (0.010227) and kNN label agreement (62.08%) in this run. The silhouette values are all close to zero, so class separation in these pooled frozen vectors appears weak.
2. Data2Vec Audio Base and HuBERT Base were substantially faster in this run, while showing lower kNN agreement than Wav2Vec2 Base.
3. Wav2Vec2 Base runtime (193.77 minutes) is unusually long relative to the other encoders. Check the run log, cache state, and runtime conditions before treating timing as a reliable model-to-model comparison.
4. These are exploratory feature-space diagnostics, not downstream classification results. Model selection should be validated with a fair frozen-feature linear probe trained on the **Train** split and evaluated on **Dev**. Do not train the classifier on Dev. If Dev is used repeatedly to choose models or tune settings, keep a separate held-out test set for the final unbiased evaluation.
5. This report records the summary metrics only. Cached embeddings and the private dataset are not included in the repository.

## Recommended next step

Run a consistent frozen-feature linear-probe comparison on the Train split and evaluate each probe on Dev, initially prioritizing Wav2Vec2 Base, HuBERT Base, and Data2Vec Audio Base. Keep the same data subset, classifier, preprocessing, and evaluation settings across candidates.
