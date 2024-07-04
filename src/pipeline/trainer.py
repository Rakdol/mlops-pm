A = [[2, 3], [3, 4]]
B = [[1, 0], [1, 2]]
C = [[6, 5], [8, 7]]


# rnadom n x 1 vector
import random

r_vector = [random.randint(0, 1) for _ in range(2)]


# r_vector = [1, 1]
def matrix_vector_product(mat, vec):
    res = []
    for b_e in B:
        t = 0
        for e, r in zip(b_e, r_vector):
            t += e * r
        res.append(t)
    return res


# Calculate Br n x n X n x 1 -> n x 1
Br = matrix_vector_product(B, r_vector)
Cr = matrix_vector_product(B, r_vector)
ABr = matrix_vector_product(A, Br)

print(ABr == Cr)
