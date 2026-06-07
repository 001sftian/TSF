from __future__ import annotations

import math

Matrix = list[list[float]]
Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def matvec(a: Matrix, x: Vector) -> Vector:
    return [dot(row, x) for row in a]


def select_columns(a: Matrix, cols: tuple[int, ...]) -> Matrix:
    return [[row[c] for c in cols] for row in a]


def column_norms(a: Matrix) -> Vector:
    if not a:
        return []
    return [math.sqrt(sum(row[j] * row[j] for row in a)) for j in range(len(a[0]))]


def solve(a: Matrix, b: Vector) -> Vector:
    n = len(b)
    aug = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            aug[pivot][col] = 1e-14
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def ridge(x: Matrix, y: Vector, lam: float) -> Vector:
    if not x or not x[0]:
        return []
    p = len(x[0])
    gram = [[0.0 for _ in range(p)] for _ in range(p)]
    rhs = [0.0] * p
    for row, yy in zip(x, y):
        for i in range(p):
            rhs[i] += row[i] * yy
            for j in range(p):
                gram[i][j] += row[i] * row[j]
    for i in range(p):
        gram[i][i] += lam
    return solve(gram, rhs)
