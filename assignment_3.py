import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

clf2 = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(classification_report(yte, clf2.predict(Xte)))

clf10 = LogisticRegression(max_iter=1000).fit(X_train, y_train)   
print(classification_report(y_test, clf10.predict(X_test)))