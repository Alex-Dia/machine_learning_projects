# Deep Learning — Image and Text Classification

**Context:** Griffith University deep learning coursework (2026).

## Notebooks

### [CIFAR-10: Softmax Regression and CNNs](notebooks/cifar10-softmax-cnn.ipynb)

Uses PyTorch, torchvision and the `d2l` Python package to compare softmax regression against CNNs of multiple depths, then explore dropout/weight decay, learning rates and mini-batch sizes. The dataset is downloaded by torchvision (`download=True`) into `./data` relative to the notebook working directory.

### [IMDb: RNN, LSTM and Bidirectional RNN](notebooks/imdb-rnn-lstm-birnn.ipynb)

Uses PyTorch and Hugging Face `datasets` to obtain IMDb movie reviews, tokenize text, build a vocabulary, and train/evaluate vanilla RNN, LSTM and bidirectional RNN architectures. The code downloads the dataset using `load_dataset("imdb")` and requires network access on first run.

## Setup (macOS / Linux)

```bash
cd deep-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd notebooks
jupyter lab
```

Choose a compatible Python version for your operating system and PyTorch / torchvision / d2l stack. On Apple Silicon, check device selection in the notebook (MPS vs CPU). PyTorch and `d2l` compatibility should be verified before running; the requirements list reflects imported packages rather than a tested, pinned environment.

## Before publishing

- Edit the first markdown cell in each notebook to use a professional title; remove the student ID from the IMDb notebook and correct 'Assigntment' to 'Assignment'.
- Explain model configurations and the key experimental findings in your own words, using the notebook's actual logged results.
- Review outputs and metadata for personal information, local paths or excessive training logs before publishing.
- The CIFAR-10 experiment includes numerous training runs (including batch size 1); a complete re-execution can be computationally expensive.
- Optionally preserve selected figures/tables as notebook outputs for a readable GitHub preview, while removing excessive logs and transient errors.
