"""Метод прогонки, п. 1.1.2 сборника задач."""
from math import isfinite
from numerics import residual


def get_example():
    # Пример 10: a[0] = c[-1] = 0, остальные нули также допустимы.
    return ([0, 6, -3, -9, -5], [-7, 12, 5, 21, -6],
            [-6, 0, 0, 8, 0], [-75, 126, 13, -40, -24])


def sweep(a, b, c, d):
    n = len(b)
    if not n or any(len(v) != n for v in (a, c, d)):
        raise ValueError("Четыре вектора должны иметь одинаковую положительную длину.")
    if any(not isfinite(x) for v in (a, b, c, d) for x in v):
        raise ValueError("Нужны конечные числа.")
    if a[0] != 0 or c[-1] != 0:
        raise ValueError("Требуется a[0] = c[-1] = 0.")
    p, q = [0.0] * n, [0.0] * n
    for i in range(n):
        denominator = b[i] + (a[i] * p[i - 1] if i else 0)
        if denominator == 0:
            raise ValueError(f"Прогонка неприменима: нулевой знаменатель на шаге {i + 1}.")
        p[i] = -c[i] / denominator
        q[i] = (d[i] - (a[i] * q[i - 1] if i else 0)) / denominator
    x = q[:]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x, p, q


def dense(a, b, c):
    n = len(b)
    return [[b[i] if j == i else a[i] if j == i - 1 else c[i] if j == i + 1 else 0
             for j in range(n)] for i in range(n)]


def main(example=False, eps=1e-6):
    if example:
        a, b, c, d = get_example()
    else:
        n = int(input("Размерность N: "))
        if n < 1:
            raise ValueError("Размерность должна быть положительной.")
        from lab1_1 import read_numbers
        a = [0.0] + (read_numbers("Поддиагональ (N-1 чисел): ", n - 1) if n > 1 else [])
        b = read_numbers("Главная диагональ: ", n)
        c = (read_numbers("Наддиагональ (N-1 чисел): ", n - 1) if n > 1 else []) + [0.0]
        d = read_numbers("Правая часть: ", n)
    x, p, q = sweep(a, b, c, d)
    print("P =", p, "\nQ =", q, "\nx =", x)
    print("Невязка:", residual(dense(a, b, c), x, d))
    return 0
