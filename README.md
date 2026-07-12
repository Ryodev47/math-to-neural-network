# math-to-neural-network

A neural network built **from scratch using only NumPy for the underlying math** — no deep learning framework (no TensorFlow, no PyTorch, no autograd). NumPy handles the low-level array/matrix operations, but every algorithm — forward propagation, backpropagation, gradient descent — is derived and implemented manually, without relying on any built-in layers, optimizers, or automatic differentiation.

This project was built as part of my *Karya Tulis Ilmiah* (KTI) / school research paper, exploring the mathematical foundations behind how neural networks actually learn.

---

## Overview

The goal of this project is to demystify neural networks by implementing one from the ground up — translating the underlying linear algebra and calculus (matrix multiplication, chain rule, gradient descent) directly into working code, rather than relying on a deep learning framework that hides the math and automates it (e.g. autograd, built-in layers/optimizers).

The network is trained to recognize handwritten digits (0–9) using the classic **MNIST** dataset.

### What's implemented from scratch
- Weight & bias initialization
- Forward propagation (matrix operations)
- ReLU and Softmax activation functions
- One-hot encoding of labels
- Backpropagation (chain rule, derived manually)
- Gradient descent parameter updates
- Accuracy evaluation

## Architecture

A simple Multi-Layer Perceptron (MLP) with:

```
Input Layer   : 784 neurons  (28x28 pixel image, flattened)
Hidden Layer 1: 16 neurons   (ReLU activation)
Hidden Layer 2: 16 neurons   (ReLU activation)
Output Layer  : 10 neurons   (Softmax activation, digits 0-9)
```

**Forward propagation:**

```
Z1 = W1 . X  + b1        A1 = ReLU(Z1)
Z2 = W2 . A1 + b2        A2 = ReLU(Z2)
Z3 = W3 . A2 + b3        A3 = Softmax(Z3)
```

**Backpropagation** uses the chain rule to compute gradients layer by layer, starting from the output layer's error (prediction − true label, via one-hot encoding) and propagating it backward through each hidden layer.

## Dataset

This project uses the [MNIST Digit Recognizer dataset from Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data) (`train.csv`): 42,000 rows, each representing a 28x28 grayscale image (784 pixel values) plus a label column (the correct digit).

The dataset is **not included** in this repository (to keep it lightweight). To run this project:

1. Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data).
2. Place it inside the `data/` folder in the project root (see `data/DATASET.md`):
   ```
   math-to-neural-network/
   └── data/
       └── train.csv
   ```
3. Update the CSV path in `notebooks/main.ipynb` if needed:
   ```python
   df = pd.read_csv('data/train.csv')
   ```

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- jupyterlab

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Ryodev47/math-to-neural-network.git
   cd math-to-neural-network
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (see [Dataset](#dataset) above) and place `train.csv` inside `data/`.
4. Open and run the notebook:
   ```bash
   jupyter lab notebooks/main.ipynb
   ```

## Results

The network is trained for 600 epochs with a learning rate (`alpha`) of 0.1. Training progress (accuracy every 50 epochs) is printed during training, and final train/test accuracy is reported at the end of the notebook.

Accuracy per training epochs:
> Epoch: 0 to 50
> Accuracy: 0.1224
> Epoch: 50 to 100
> Accuracy: 0.5463
> Epoch: 100 to 150
> Accuracy: 0.6891
> Epoch: 150 to 200
> Accuracy: 0.7678
> Epoch: 200 to 250
> Accuracy: 0.8067
> Epoch: 250 to 300
> Accuracy: 0.8275
> Epoch: 300 to 350
> Accuracy: 0.8411
> Epoch: 350 to 400
> Accuracy: 0.8509
> Epoch: 400 to 450
> Accuracy: 0.8594
> Epoch: 450 to 500
> Accuracy: 0.8665
> Epoch: 500 to 550
> Accuracy: 0.8719
> Epoch: 550 to 600
> Accuracy: 0.8770

A helper function `testPrediction()` lets you visualize a single test image alongside the model's predicted digit vs. the actual label.

Final Accuracy:
- Train Accuracy: `0.8819`
- Test Accuracy: `0.8780`

## Repository Structure

```
math-to-neural-network/
├── data/
│   └── DATASET.md          # instructions for downloading the dataset
├── notebooks/
│   └── main.ipynb          # full implementation: preprocessing, model, training, evaluation
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Notes & Learnings

This project was built to deeply understand:
- How matrix operations map to a neural network's forward pass
- How the chain rule is applied layer-by-layer during backpropagation
- Why gradients are averaged over batch size for stable training
- How softmax + one-hot encoding work together for multi-class classification

### Reproducibility (random seed)

The notebook sets a fixed random seed (`np.random.seed(4)`) before splitting the data and initializing the network's weights and biases. Both the train/test split and the weight initialization rely on NumPy's random number generator, which is normally non-deterministic (different every run).

By fixing the seed, the same sequence of "random" numbers is generated every time the notebook runs — so the train/test split stays the same, the initial weights stay the same, and as a result, **training and test accuracy stay consistent across runs** instead of fluctuating each time due to a different random initialization or data split. This makes it easier to debug the model and to report accuracy numbers that others can actually reproduce.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [MNIST Digit Recognizer — Kaggle](https://www.kaggle.com/competitions/digit-recognizer)
- Built for educational purposes as part of a school research paper (KTI)
