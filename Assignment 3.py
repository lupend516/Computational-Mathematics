# Task 1: Iterative Method for Matrix Inversion
import numpy as np

def iterative_matrix_inversion(A, tol=1e-6):
    n = A.shape[0]
    B = np.eye(n) / np.trace(A)  
    I = np.eye(n)
    
    for _ in range(1000):  
        R = I - A @ B
        B_next = B + B @ R
        if np.linalg.norm(R, ord=np.inf) < tol:
            break
        B = B_next

    return B

A = np.array([[4, 7], [2, 6]])  
inv_A_iterative = iterative_matrix_inversion(A)
inv_A_numpy = np.linalg.inv(A)

print("Inverse matrix (iterative):\n", inv_A_iterative)
print("Inverse matrix (numpy):\n", inv_A_numpy)

# Task 2: LU Factorization
from scipy.linalg import lu

A = np.array([[4, 3], [6, 3]])  
b = np.array([10, 12])  

P, L, U = lu(A)
x = np.linalg.solve(A, b)

print("L matrix:\n", L)
print("U matrix:\n", U)
print("Solution x (LU):", x)

# Task 3: Largest Eigenvalue and Vector (Power Method)
def power_method(A, v0, tol=1e-6, max_iter=1000):
    v = v0 / np.linalg.norm(v0)
    for _ in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        if np.linalg.norm(v_new - v, ord=np.inf) < tol:
            break
        v = v_new

    eigenvalue = np.dot(v.T, A @ v)
    return eigenvalue, v

A = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])  
v0 = np.array([1, 0, 0])

eigenvalue, eigenvector = power_method(A, v0)
numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)

print("Largest eigenvalue (Power Method):", eigenvalue)
print("Largest eigenvector (Power Method):", eigenvector)
print("Eigenvalues (numpy):", numpy_eigenvalues)

# Task 4: Givens and Householder Reduction
from scipy.linalg import qr

A = np.array([[4, 3], [6, 3]])  
Q, R = qr(A, mode='full')

print("Q matrix (Householder):\n", Q)
print("R matrix (Householder):\n", R)



# Task 5: Jacobi Method for All Eigenvalues
def jacobi_method(A, tol=1e-6):
    def max_offdiag(A):
        n = A.shape[0]
        max_val = 0
        k, l = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    k, l = i, j
        return max_val, k, l

    n = A.shape[0]
    V = np.eye(n)
    while True:
        max_val, k, l = max_offdiag(A)
        if max_val < tol:
            break

        theta = 0.5 * np.arctan2(2 * A[k, l], A[k, k] - A[l, l])
        c, s = np.cos(theta), np.sin(theta)

        J = np.eye(n)
        J[k, k], J[l, l] = c, c
        J[k, l], J[l, k] = s, -s

        A = J.T @ A @ J
        V = V @ J

    eigenvalues = np.diag(A)
    return eigenvalues, V

A = np.array([[4, 2], [2, 3]])  
eigenvalues, eigenvectors = jacobi_method(A)
numpy_eigenvalues = np.linalg.eigvals(A)

print("Eigenvalues (Jacobi):", eigenvalues)
print("Eigenvalues (numpy):", numpy_eigenvalues)
