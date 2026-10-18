from math import isfinite


def validate_eps(eps):
    if not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть положительным конечным числом.")


def read_eps():
    while True:
        try:
            eps = float(input("Точность вычислений eps: "))
            validate_eps(eps)
            return eps
        except ValueError as error:
            print(error)


def norm(x):
    return max(abs(value) for value in x)
