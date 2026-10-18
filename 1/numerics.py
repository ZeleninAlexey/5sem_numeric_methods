"""Общие операции над матрицами без сторонних библиотек."""
from math import isfinite
from lab1_1 import parse_matrix_from_console, parse_vector_from_console, print_matrix


def validate_matrix(a):
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("Нужна непустая квадратная матрица.")
    if any(not isfinite(v) for row in a for v in row):
        raise ValueError("Элементы должны быть конечными числами.")
    return n


def validate_eps(eps):
    if not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть положительным конечным числом.")


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def norm(v):
    return max((abs(x) for x in v), default=0.0)


def residual(a, x, b):
    return norm([sum(v * y for v, y in zip(row, x)) - rhs for row, rhs in zip(a, b)])


def input_system():
    n = int(input("Размерность N: "))
    if n < 1:
        raise ValueError("Размерность должна быть положительной.")
    return parse_matrix_from_console(n), parse_vector_from_console(n)


def input_matrix():
    n = int(input("Размерность N: "))
    if n < 1:
        raise ValueError("Размерность должна быть положительной.")
    return parse_matrix_from_console(n)
