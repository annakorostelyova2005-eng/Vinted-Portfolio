# Python TRM Project

## Project overview

This project is an experimental implementation and evaluation of a Tiny Recursive Model (TRM) for tabular prediction tasks.

The main goal is to compare whether recursive computation improves performance compared with simpler baselines. The project tests TRM-style recursion against:
- a TRM version with no recursion,
- a parameter-matched MLP baseline.

The experiments cover both regression and classification settings, using synthetic datasets and standard scikit-learn datasets.

## What is included

| File | Description |
|---|---|
| `TRM_experiments.ipynb` | Main notebook with experiments, results, learning curves, and conclusions. |
| `trm.py` | Main TRM model implementation with recursive latent states and halting logic. |
| `trm_all_grad.py` | Alternative TRM implementation where gradients are tracked through all recursive updates. |
| `single_recursion.py` | Core recurrence engine for TRM reasoning and solution-state updates. |
| `single_recursion_all_grad.py` | Recurrence engine variant with full gradient tracking. |
| `update_network.py` | Shared computational block used repeatedly inside the recursive model. |
| `basic_mlp.py` | MLP baseline used for comparison with the TRM models. |
| `model_training.py` | Training and validation loops for TRM and MLP models. |
| `helpers.py` | Helper functions for batching, metrics, logging, and plotting. |
| `experiments_log.csv` | Logged experiment results. |
| `readme.txt` | Original short file description. |

## Experiments

The notebook compares models on:

- synthetic regression,
- real regression,
- synthetic binary classification,
- real binary classification,
- synthetic multi-class classification,
- real multi-class classification.

For each setting, the project evaluates:
- full TRM,
- TRM without recursion,
- MLP baseline.

The notebook also tracks learning curves and records experiment summaries in `experiments_log.csv`.

## How to run

1. Install the required Python packages:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

2. Open the notebook:

```bash
jupyter notebook TRM_experiments.ipynb
```

3. Run the notebook cells from top to bottom.

The project does not require a separate external dataset for the included experiments. Some datasets are generated synthetically, and the real datasets are loaded from scikit-learn.

## Notes about reproducibility

The notebook contains many experiment cells and may take some time to run, especially without a GPU. Results may vary slightly between runs because neural network training can be sensitive to random initialization and hardware.

No large model files, cache folders, or external datasets are included in this upload. The project is kept compact so it can fit within a small application-upload limit while still showing the code, model structure, experiments, and results.

## Skills demonstrated

- PyTorch model implementation
- recursive neural network architecture
- training and validation loops
- regression and classification experiments
- baseline comparison
- experiment logging
- notebook-based analysis and visualization
