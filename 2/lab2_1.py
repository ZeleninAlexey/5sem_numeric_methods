from math import cos, isfinite, sin, sqrt
from numerics import validate_eps


def f(x):
    return sin(x) - 2 * x * x + 0.5


def derivative(x):
    return cos(x) - 4 * x


def phi(x):
    return sqrt((sin(x) + 0.5) / 2)


def simple_iteration(eps=1e-6, x0=0.75, max_iterations=10000):
    validate_eps(eps)
    if not 0.7 <= x0 <= 0.8:
        raise ValueError("Начальное приближение должно лежать в [0.7, 0.8].")
    q = cos(0.7) / (4 * phi(0.7))
    x = x0
    history = []
    for step in range(1, max_iterations + 1):
        next_x = phi(x)
        error = q / (1 - q) * abs(next_x - x)
        history.append((step, next_x, error, abs(f(next_x))))
        x = next_x
        if error <= eps:
            return x, history, q
    raise ValueError("Превышено максимальное число итераций.")


def newton(eps=1e-6, x0=0.8, max_iterations=10000):
    validate_eps(eps)
    if not 0.7 <= x0 <= 0.8 or f(x0) * (-sin(x0) - 4) < 0:
        raise ValueError("Нужно x0 из [0.7, 0.8] с f(x0) * f''(x0) >= 0.")
    x = x0
    history = []
    for step in range(1, max_iterations + 1):
        df = derivative(x)
        if df == 0:
            raise ValueError("Производная равна нулю.")
        next_x = x - f(x) / df
        if not isfinite(next_x):
            raise ValueError("Получено неконечное приближение.")
        difference = abs(next_x - x)
        history.append((step, next_x, difference, abs(f(next_x))))
        x = next_x
        if difference <= eps:
            return x, history
    raise ValueError("Превышено максимальное число итераций.")


def main(eps=1e-6):
    print("sin(x) - 2*x^2 + 0.5 = 0, положительный корень")
    x, history, q = simple_iteration(eps)
    print(f"\nПростые итерации: x0 = 0.75, q = {q:.12g}")
    print(" k             x             оценка ошибки       |f(x)|")
    for step, value, error, residual in history:
        print(f"{step:2d} {value:17.12f} {error:18.6e} {residual:14.6e}")
    print(f"x = {x:.12f}, итераций: {len(history)}")
    x, history = newton(eps)
    print("\nМетод Ньютона: x0 = 0.8")
    print(" k             x             |x(k)-x(k-1)|       |f(x)|")
    for step, value, difference, residual in history:
        print(f"{step:2d} {value:17.12f} {difference:18.6e} {residual:14.6e}")
    print(f"x = {x:.12f}, итераций: {len(history)}")
    print("\nТочность       Простые итерации    Ньютон")
    for accuracy in (1e-2, 1e-4, 1e-6, 1e-8):
        print(f"{accuracy:.0e} {len(simple_iteration(accuracy)[1]):20d} {len(newton(accuracy)[1]):9d}")
