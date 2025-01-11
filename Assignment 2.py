# Task 1: Jacobi Method and Convergence Analysis

import numpy as np

A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]], dtype=float)
b = np.array([-1, 2, 3], dtype=float)

x = np.zeros_like(b)
accuracy = 1e-5

diag_dominant = np.all(2 * np.abs(np.diag(A)) > np.sum(np.abs(A), axis=1))
print("Diagonal dominance:", diag_dominant)

max_iterations = 1000
for iteration in range(max_iterations):
    x_new = np.zeros_like(x)
    for i in range(len(A)):
        s = np.dot(A[i, :], x) - A[i, i] * x[i]
        x_new[i] = (b[i] - s) / A[i, i]
    if np.linalg.norm(x_new - x, ord=np.inf) < accuracy:
        break
    x = x_new

print(f"Solution: {x}")
print(f"Iterations: {iteration + 1}")

# Task 2: Gaussian Method with Choice of Leading Element

A = np.array([[2, -1, 1], [1, 3, 2], [1, -1, 2]], dtype=float)
b = np.array([2, 12, 5], dtype=float)

n = len(b)
for i in range(n):
    max_row = i + np.argmax(np.abs(A[i:, i]))
    A[[i, max_row]] = A[[max_row, i]]
    b[[i, max_row]] = b[[max_row, i]]
    for j in range(i + 1, n):
        factor = A[j, i] / A[i, i]
        A[j, i:] -= factor * A[i, i:]
        b[j] -= factor * b[i]

x = np.zeros_like(b)
for i in range(n - 1, -1, -1):
    x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

print("Upper triangular matrix A:")
print(A)
print("Solution vector x:")
print(x)

# Task 3: Gauss-Jordan Method

A = np.array([[2, -1, 1], [1, 3, 2], [1, -1, 2]], dtype=float)
b = np.array([2, 12, 5], dtype=float)

augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
n = len(b)

for i in range(n):
    augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
    for j in range(n):
        if i != j:
            augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

x = augmented_matrix[:, -1]

print("Diagonal matrix:")
print(augmented_matrix)
print("Solution vector x:")
print(x)

# Task 4: Gauss-Seidel Method

A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]], dtype=float)
b = np.array([-1, 2, 3], dtype=float)

x = np.zeros_like(b)
accuracy = 1e-5
max_iterations = 1000

for iteration in range(max_iterations):
    x_new = np.copy(x)
    for i in range(len(A)):
        s = np.dot(A[i, :], x_new) - A[i, i] * x_new[i]
        x_new[i] = (b[i] - s) / A[i, i]
    if np.linalg.norm(x_new - x, ord=np.inf) < accuracy:
        break
    x = x_new

print(f"Solution: {x}")
print(f"Iterations: {iteration + 1}")

# Task 5: Relaxation Method

A = np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]], dtype=float)
b = np.array([-1, 2, 3], dtype=float)

omega_values = [1.1, 1.5]
results = {}

for omega in omega_values:
    x = np.zeros_like(b)
    for iteration in range(max_iterations):
        x_new = np.copy(x)
        for i in range(len(A)):
            s = np.dot(A[i, :], x_new) - A[i, i] * x_new[i]
            x_new[i] = (1 - omega) * x_new[i] + omega * (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < accuracy:
            break
        x = x_new
    results[omega] = (x, iteration + 1)

for omega, (solution, iterations) in results.items():
    print(f"Solution for omega={omega}: {solution}, Iterations: {iterations}")

# Task 6: Ill-Conditioned Systems

A = np.array([[1, 1], [1, 1.0001]], dtype=float)
b = np.array([2, 2.0001], dtype=float)

x_exact = np.linalg.solve(A, b)
print("Exact solution:", x_exact)

A_perturbed = A.copy()
A_perturbed[1, 1] += 0.0001
x_perturbed = np.linalg.solve(A_perturbed, b)
print("Solution for perturbed system:", x_perturbed)
