"""Простые итерации и метод Зейделя, п. 1.1.4 сборника."""
from math import isfinite
from numerics import validate_matrix, validate_eps, norm, residual, input_system


def get_example():
    return ([[28, 9, -3, -7], [-5, 21, -5, -3],
             [-8, 1, -16, 5], [0, -2, 5, 8]], [-159, 63, -45, 24])


def iterate(a, b, eps=1e-6, seidel=False, max_iterations=100000):
    n = validate_matrix(a)
    validate_eps(eps)
    if len(b) != n or any(not isfinite(v) for v in b):
        raise ValueError("Некорректный вектор правой части.")
    if any(a[i][i] == 0 for i in range(n)):
        raise ValueError("На диагонали есть нуль: переставьте уравнения.")
    alpha = [[-a[i][j] / a[i][i] if i != j else 0.0 for j in range(n)] for i in range(n)]
    beta = [b[i] / a[i][i] for i in range(n)]
    # Для Зейделя строится матрица перехода T=(E-B)^(-1)C прямой подстановкой.
    transition = [row[:] for row in alpha]
    if seidel:
        transition = [[0.0] * n for _ in range(n)]
        for j in range(n):
            for i in range(n):
                transition[i][j] = (alpha[i][j] if j > i else 0) + sum(
                    alpha[i][k] * transition[k][j] for k in range(i))
    contraction = max(sum(abs(v) for v in row) for row in transition)
    if contraction >= 1:
        raise ValueError("Условие сходимости в бесконечной норме не выполнено: ||T|| >= 1.")
    x = beta[:]
    history = []
    for step in range(1, max_iterations + 1):
        previous = x[:]
        for i in range(n):
            x[i] = beta[i] + sum(alpha[i][j] * (x[j] if seidel and j < i else previous[j])
                                 for j in range(n))
        error = contraction / (1 - contraction) * norm([x[i] - previous[i] for i in range(n)])
        if not all(isfinite(v) for v in x):
            raise ValueError("Итерации привели к переполнению.")
        history.append((step, error, residual(a, x, b)))
        if error <= eps:
            return x, history, contraction
    raise ValueError("Превышено максимальное число итераций.")


def main(example=False, eps=1e-6):
    a, b = get_example() if example else input_system()
    for seidel, label in [(False, "Простые итерации"), (True, "Зейдель")]:
        x, history, q = iterate(a, b, eps, seidel)
        print(label, "x =", x, "итераций:", len(history), "||T|| =", q)
        print("Оценка погрешности:", history[-1][1], "невязка:", history[-1][2])
    print("Зависимость числа итераций от точности:")
    for accuracy in (1e-2, 1e-4, 1e-6, 1e-8):
        print(accuracy, *[len(iterate(a, b, accuracy, method)[1]) for method in (False, True)])
    return 0
