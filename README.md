# SBED

Implementation for **Efficient Signed Bipartite Graph Learning via Equilibrium Dynamics**.

SBED is a framework for link-sign prediction on signed bipartite graphs. Given an observed signed training graph and a candidate interaction between its two node partitions, SBED estimates the probability that the interaction is positive.

## Overview

SBED combines four components:

1. **Signed bipartite spectral encoding:** Truncated SVD of the signed biadjacency matrix provides partition-specific structural features, which are processed by two type-specific feature encoders.
2. **Bipartite equilibrium modeling:** Friedkin–Johnsen dynamics describe global signed influence through cross-partition odd-parity and same-partition even-parity channels.
3. **Partition-Aware Expansion Forest Sampling (PA-EFS):** Rooted-forest sampling on the unsigned double-cover graph approximates equilibrium influence, with separate retention budgets for the two channels.
4. **Parity-aware aggregation and equilibrium-aware decoding:** Adaptive aggregation combines the retained equilibrium evidence, and the decoder predicts interaction signs.

The sparse equilibrium representation avoids explicitly forming or inverting the dense equilibrium matrix.

## Reference Software Environment

The following software configuration is provided as a reference based on the [SigFJProp README](https://github.com/SunYubo23/SigFJProp/blob/master/README.md).

| Software | Reference version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 2.3.0 |
| NumPy | 1.23 |
| pandas | 1.5.3 |
| SciPy | 1.13 |
| scikit-learn | 1.6.1 |
| tqdm | 4.67.1 |
| texttable | 1.7.0 |

The example torch-scatter wheel provided by SigFJProp targets Python 3.9, PyTorch 2.3, and CUDA 11.8 on 64-bit Windows. These versions describe the reference configuration; the environment used for the SBED experiments must be verified separately.

## Datasets

We evaluate SBED on six signed bipartite datasets. The statistics below correspond to Table 1 of the manuscript.

| Dataset | First partition | Second partition | Signed interactions | Positive (%) | Negative (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Review | 182 | 304 | 1,170 | 39.66 | 60.34 |
| Senate | 145 | 1,056 | 27,083 | 55.31 | 44.69 |
| Bonanza | 7,919 | 1,973 | 36,543 | 97.98 | 2.02 |
| House1to10 | 515 | 1,281 | 114,378 | 53.96 | 46.04 |
| MovieLens-1M | 6,040 | 3,706 | 1,000,209 | 57.52 | 42.48 |
| Amazon-Book | 35,736 | 38,121 | 1,960,674 | 80.58 | 19.42 |

The interaction signs are defined as follows:

- **Review:** Strong accept, accept, and weak accept are positive; weak reject, reject, and strong reject are negative.
- **Senate and House1to10:** Yea votes are positive and Nay votes are negative.
- **Bonanza:** Signs correspond to positive and negative buyer–seller feedback.
- **MovieLens-1M and Amazon-Book:** Ratings of 4 or 5 are positive; ratings of 1, 2, or 3 are negative.


## Experimental Protocol

### Data splitting

Positive and negative interactions are stratified into training, validation, and test sets at a ratio of **80:10:10**.

We adopt a transductive evaluation setting and ensure that the endpoints of validation and test interactions occur in the training graph.

### Structural preprocessing

Only signed training interactions are used to:

- Construct the signed biadjacency matrix.
- Compute the truncated SVD.
- Construct the unsigned double-cover graph.
- Sample rooted forests.
- Select partition-aware equilibrium neighborhoods.
- Compute training class weights.

Signed training interactions provide both structural context and supervision. For each training split and sampling seed, preprocessing outputs are computed once before neural optimization and remain fixed during training.

Validation and test interactions, including their signs, are excluded from structural preprocessing.

### Model selection and evaluation

Hyperparameters and checkpoints are selected using **validation AUC**. Test labels are used only for final evaluation.

We report mean performance over **five independent runs** with different random seeds. Compared methods use the same data splits, metric definitions, and evaluation procedures.

The evaluation metrics are:

- **AUC**
- **Binary-F1**, treating positive interactions as the positive class
- **Macro-F1**
- **Micro-F1**

AUC and Macro-F1 are the primary metrics because several datasets exhibit substantial sign imbalance.

## Parameters

### Shared settings

| Parameter | Meaning | Setting |
| --- | --- | ---: |
| `d` | Output embedding dimension | 32 |
| Encoder depth | Layers in each type-specific MLP | 2 |
| Activation | Encoder activation | ReLU |
| `S` | Number of sampled forests | 400 |
| `K_c` | Cross-partition off-diagonal retention budget | 16 |
| `K_s` | Same-partition off-diagonal retention budget | 16 |
| `K` | Total retention budget, `K_c + K_s` | 32 |
| `rho` | Cross-partition budget ratio, `K_c / K` | 0.50 |
| Optimizer | Neural optimization | Adam |
| Loss | Classification objective | Class-weighted binary cross-entropy |
| Batch size | Interactions per decoder batch | 2,048 |
| Maximum epochs | Training limit | 500 |
| Early-stopping patience | Epochs without improvement in validation AUC | 30 |

These symbols follow the manuscript. Their corresponding command-line arguments or configuration keys follow the released implementation.

### Dataset-specific settings

The following configurations correspond to Table C2 of the manuscript. All datasets use the shared sampling and neighborhood settings above.

| Dataset | SVD rank | Learning rate | Weight decay | Dropout |
| --- | ---: | ---: | ---: | ---: |
| Review | 16 | 0.001 | 0.0001 | 0.2 |
| Senate | 32 | 0.001 | 0.00001 | 0.1 |
| Bonanza | 64 | 0.0005 | 0.0001 | 0.2 |
| House1to10 | 64 | 0.001 | 0.00001 | 0.2 |
| MovieLens-1M | 64 | 0.005 | 0.00001 | 0.1 |
| Amazon-Book | 128 | 0.001 | 0.00001 | 0.2 |

Hyperparameters are selected using validation AUC. The search spaces are provided in Appendix C of the manuscript.

## Running the Code

### Download the repository

```text
[TO_FILL: repository URL and clone command]
```

### Train and evaluate on one dataset

```text
[TO_FILL: exact command for a single run, including the dataset, configuration, seed, and output location]
```

### Reproduce the five-run evaluation

```text
[TO_FILL: exact command or script for all five runs]
```


## Reproducibility

### Random seeds

The five random seeds used in the reported experiments are:

```text
[TO_FILL: the five random seeds actually used]
```

When controlled separately, the split seed, forest-sampling seed, and neural-training seed should be recorded in the experiment configuration.

SBED and its ablation variants use matched data splits and random seeds. Configurations in the performance–runtime trade-off analysis also use matched splits and seeds.

### Experiment files

```text
[TO_FILL: paths or download links for saved splits, configurations, run logs, and checkpoints]
```

### Runtime measurements

We distinguish one-time structural preprocessing, per-epoch training, test inference, and total runtime.

- **Figure 5:** Reports average training time per epoch and inference time per 100,000 candidate interactions. One-time structural preprocessing is excluded.
- **Figure 7:** Reports total runtime, including one-time graph-dependent preprocessing, training and validation until early stopping, and inference on all test interactions. Data loading and disk I/O are excluded. Total runtime and test AUC are averaged over five runs.

## Citation

If you use SBED in your research, please cite the manuscript:

```bibtex
@unpublished{wang2026sbed,
  title  = {Efficient Signed Bipartite Graph Learning via Equilibrium Dynamics},
  author = {Wang, Xiaosong and Gu, Lichuan},
  year   = {2026},
  note   = {Manuscript}
}
```

## Acknowledgements

We thank the authors of [SigFJProp](https://github.com/SunYubo23/SigFJProp) for making their implementation publicly available.

## Contact

For questions about this work, please contact Lichuan Gu at [glc@ahau.edu.cn](mailto:glc@ahau.edu.cn).
