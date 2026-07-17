# nano-tabpfn

A from-scratch build of [TabPFN](https://www.nature.com/articles/s41586-024-08328-6.pdf) — the data priors, the attention-based architecture, and the training loop — done to actually understand how a foundation model for tabular data works under the hood, rather than just calling `.fit()` on someone else's checkpoint.

## Why

Most tabular ML (XGBoost, Random Forest, etc.) fits a model to `X_train, y_train` and then runs inference on `X_test` in a separate phase. TabPFN collapses both into a single forward pass: `X_train`, `y_train`, and `X_test` are fed in together like a prompt, and the model predicts `y_test` directly through in-context learning — the same trick that makes LLMs able to solve new tasks from a few examples, except here the "tokens" are cells in a table. It gets this ability by pretraining on millions of synthetic datasets sampled from a prior instead of being fit per-dataset.

I wanted to build this end to end myself: write the priors that generate synthetic tables, implement the row/feature attention architecture, train it, and see how close a tiny version gets to the real thing.

## What's in here

1. **Linear prior** — the simplest possible synthetic data generator (`y = mx + c + noise`), used to validate the architecture before scaling up to anything harder.
2. **Architecture** — a `FeatureEncoder` and `TargetEncoder` that normalize and embed table cells, a `TransformerEncoderLayer` that alternates attention between features and attention between datapoints (with masking so test rows never leak into training statistics or attend to each other), and a `Decoder` that turns embeddings into predictions. Wired together as `NanoTabPFNModel`.
3. **Training loop** — schedule-free AdamW plus a bar-distribution loss, first tried on the linear prior.
4. **MLP-SCM prior** — a much richer synthetic data generator built from random structural causal models (small randomly-initialized MLPs with random activations), used to generate diverse classification datasets on the fly and train a real classifier.
5. **Head-to-head comparison** — the trained model against SVM, Logistic Regression, Random Forest, and the official TabPFN v2 on toy 2D classification tasks (moons, circles, linearly separable data).

Getting this right meant chasing down some genuinely subtle bugs along the way — mostly data leakage: normalizing features using test-set statistics, letting training rows attend to test rows, standardizing columns with a global mean/std instead of per-column, padding features before permuting them, and invalid train/test class splits. Each fix is called out inline in the notebook.

## Requirements

- Python 3.11.13
- `torch==2.6.0`
- `pfns==0.3.0`
- `tabpfn==2.0.9`
- `numpy`, `scikit-learn`, `matplotlib`, `schedulefree`, `h5py`, `tqdm`

```bash
pip install torch==2.6.0 pfns==0.3.0 tabpfn==2.0.9 numpy scikit-learn matplotlib schedulefree h5py tqdm
```

## Running it

Open `nano-tabpfn-exercise-v2.ipynb` and run all cells top to bottom. Training is kept small on purpose (a few minutes on CPU) — bump up `batch_size`, `num_steps`, `epochs`, `max_features`, or `max_seq_len` for a stronger model if you've got a GPU handy.

## Results

On the toy 2D datasets, the resulting model produces decision boundaries broadly comparable to the classical baselines and qualitatively similar to official TabPFN v2 — despite training on a much simpler prior for a fraction of the compute. The notebook includes a written comparison across TabPFN, SVM, Logistic Regression, and Random Forest.

## Credit

Built on top of [nanoTabPFN](https://github.com/automl/nanoTabPFN) by Pfefferle, Hog, Purucker, and Hutter ([paper](https://arxiv.org/pdf/2511.03634)).
