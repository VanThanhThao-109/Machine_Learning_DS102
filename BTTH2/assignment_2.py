import os
import numpy as np
import matplotlib.pyplot as plt 

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MNIST dataset")

def load_images(name):
    return np.fromfile(os.path.join(BASE, name), dtype=np.uint8, offset=16).reshape(-1, 784)

def load_labels(name):
    return np.fromfile(os.path.join(BASE, name), dtype=np.uint8, offset=8)

X_train = load_images("train-images.idx3-ubyte") / 255.0
y_train = load_labels("train-labels.idx1-ubyte").astype(int)
X_test  = load_images("t10k-images.idx3-ubyte") / 255.0
y_test  = load_labels("t10k-labels.idx1-ubyte").astype(int)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z); return E / E.sum(axis=1, keepdims=True)

C, d = 10, X_train.shape[1]; N = len(y_train)
W = np.zeros((d, C)); b = np.zeros(C)
Y = np.eye(C)[y_train]                       
lr, epochs, eps = 0.5, 300, 1e-12
losses = []
for ep in range(epochs):
    P = softmax(X_train @ W + b)
    losses.append(-np.mean(np.sum(Y * np.log(P + eps), axis=1)))
    G = (P - Y) / N                          
    W -= lr * (X_train.T @ G)               
    b -= lr * G.sum(axis=0)

def prf1(y_true, y_pred, cls):                 
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    p = tp / (tp + fp + 1e-12); r = tp / (tp + fn + 1e-12)
    return p, r, 2 * p * r / (p + r + 1e-12)

pred = softmax(X_test @ W + b).argmax(axis=1)
res = [prf1(y_test, pred, c) for c in range(C)]
for c, (p, r, f) in enumerate(res): print(f"digit {c}: P={p:.3f} R={r:.3f} F1={f:.3f}")
print("macro:", np.mean(res, axis=0), "| accuracy:", (pred == y_test).mean())

plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
plt.title("Softmax Regression: training loss"); plt.show()

