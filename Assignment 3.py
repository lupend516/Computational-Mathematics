
import numpy as np

A = np.array([[4, 1, 2], [1, 3, 0], [2, 0, 1]], dtype=float)
trace_A = np.trace(A)
B = np.identity(A.shape[0]) / trace_A
accuracy = 1e-6
max_iterations = 1000

for i in range(max_iterations):
    B_next = B @ (2 * np.identity(A.shape[0]) - A @ B)
    if np.linalg.norm(B_next - B, ord='fro') < accuracy:
        break
    B = B_next

print("Computed Inverse Matrix:")
print(B)
print("Numpy's Inverse Matrix:")
print(np.linalg.inv(A))
