from math import cos, isfinite, sin
from numerics import norm, validate_eps


def functions(x):
    x1, x2 = x
    return (x1 - cos(x2) - 1, x2 - sin(x1) - 1)


def phi(x):
    x1, x2 = x
    return (1 + cos(x2), 1 + sin(x1))


def jacobian(x):
    x1, x2 = x
    return [[1.0, sin(x2)], [-cos(x1), 1.0]]


def validate_start(x):
    if len(x) != 2 or not (0.75 <= x[0] <= 0.9 and 1.68 <= x[1] <= 1.79):
        raise ValueError("Начальное приближение должно лежать в [0.75, 0.9] x [1.68, 1.79].")


def simple_iteration(eps=1e-6, x0=(0.8, 1.75), max_iterations=10000):
    validate_eps(eps)
    validate_start(x0)
    q = max(sin(1.68), cos(0.75))
    x = tuple(x0)
    history = []
    for step in range(1, max_iterations + 1):
        next_x = phi(x)
        error = q / (1 - q) * norm([next_x[i] - x[i] for i in range(2)])
        history.append((step, next_x, error, norm(functions(next_x))))
        x = next_x
        if error <= eps:
            return x, history, q
    raise ValueError("Превышено максимальное число итераций.")


def newton(eps=1e-6, x0=(0.8, 1.75), max_iterations=10000):
    validate_eps(eps)
    validate_start(x0)
    x = tuple(x0)
    history = []
    for step in range(1, max_iterations + 1):
        j = jacobian(x)
        f1, f2 = functions(x)
        multiplier = j[1][0] / j[0][0]
        denominator = j[1][1] - multiplier * j[0][1]
        if denominator == 0:
            raise ValueError("Матрица Якоби вырождена.")
        delta2 = (-f2 + multiplier * f1) / denominator
        delta1 = (-f1 - j[0][1] * delta2) / j[0][0]
        next_x = (x[0] + delta1, x[1] + delta2)
        if not all(isfinite(value) for value in next_x):
            raise ValueError("Получено неконечное приближение.")
        difference = norm([next_x[i] - x[i] for i in range(2)])
        history.append((step, next_x, difference, norm(functions(next_x))))
        x = next_x
        if difference <= eps:
            return x, history
    raise ValueError("Превышено максимальное число итераций.")


def main(eps=1e-6):
    print("x1 - cos(x2) = 1, x2 - sin(x1) = 1")
    print("Начальное приближение: (0.8, 1.75)")
    x, history, q = simple_iteration(eps)
    print(f"\nПростые итерации: q = {q:.12g}")
    print(" k            x1                x2          оценка ошибки      ||F(x)||")
    for step, value, error, residual in history:
        print(f"{step:2d} {value[0]:17.12f} {value[1]:17.12f} {error:16.6e} {residual:14.6e}")
    print(f"x = ({x[0]:.12f}, {x[1]:.12f}), итераций: {len(history)}")
    x, history = newton(eps)
    print("\nМетод Ньютона")
    print(" k            x1                x2          ||x(k)-x(k-1)||   ||F(x)||")
    for step, value, difference, residual in history:
        print(f"{step:2d} {value[0]:17.12f} {value[1]:17.12f} {difference:16.6e} {residual:14.6e}")
    print(f"x = ({x[0]:.12f}, {x[1]:.12f}), итераций: {len(history)}")
    print("\nТочность       Простые итерации    Ньютон")
    for accuracy in (1e-2, 1e-4, 1e-6, 1e-8):
        print(f"{accuracy:.0e} {len(simple_iteration(accuracy)[1]):20d} {len(newton(accuracy)[1]):9d}")
