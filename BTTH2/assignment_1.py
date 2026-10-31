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


m_tr, m_te = y_train < 2, y_test < 2
Xtr, ytr = X_train[m_tr], y_train[m_tr]
Xte, yte = X_test[m_te],  y_test[m_te]

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

w = np.zeros(Xtr.shape[1]); b = 0.0
lr, epochs, N, eps = 0.1, 300, len(ytr), 1e-12
losses = []
for ep in range(epochs):
    yhat = sigmoid(Xtr @ w + b)
    losses.append(-np.mean(ytr*np.log(yhat+eps) + (1-ytr)*np.log(1-yhat+eps)))
    g = yhat - ytr                       
    w -= lr * (Xtr.T @ g) / N            
    b -= lr * g.mean()
    
def prf1(y_true, y_pred, cls):                 
    tp = np.sum((y_pred == cls) & (y_true == cls))
    fp = np.sum((y_pred == cls) & (y_true != cls))
    fn = np.sum((y_pred != cls) & (y_true == cls))
    p = tp / (tp + fp + 1e-12); r = tp / (tp + fn + 1e-12)
    return p, r, 2 * p * r / (p + r + 1e-12)

pred = (sigmoid(Xte @ w + b) >= 0.5).astype(int)
print("precision, recall, f1 (lớp 1):", prf1(yte, pred, 1))

plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("BCE loss")
plt.title("Logistic Regression: training loss"); plt.show()

