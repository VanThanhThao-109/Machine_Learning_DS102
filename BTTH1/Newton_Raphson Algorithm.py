import numpy as np

def f(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2 - 3*x[0]

def grad(x):
    return np.array([
        2*x[0] + x[1] - 3,
        x[0] + 2*x[1]
    ])

def hess(x):
    return np.array([
        [2.0, 1.0],
        [1.0, 2.0]
    ])

x = np.array([0.0, 0.0])

print("Gia tri f tai diem ban dau:", f(x))
print("Gradient tai diem ban dau:", grad(x))
print("Hessian tai diem ban dau:\n", hess(x))

tol = 1e-8
max_iter = 50

for k in range(max_iter):
    g = grad(x)
    grad_norm = np.linalg.norm(g)
    print("Lan lap:", k)
    print("  Vi tri hien tai:", x)
    print("  Do dai gradient:", grad_norm)
    
    H = hess(x)
    d = np.linalg.solve(H, g)
    x = x - d

print("Nghiem tim duoc:", x)
print("Gia tri nho nhat f(x):", f(x))
print("Gradient tai nghiem:", grad(x))

eigenvalues = np.linalg.eigvalsh(hess(x))
print("Cac tri rieng cua Hessian:", eigenvalues)

