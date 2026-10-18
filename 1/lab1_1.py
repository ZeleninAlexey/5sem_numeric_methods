"""Метод Гаусса по п. 1.1.1 сборника задач У. Г. Пирумова."""

from math import isfinite
import sys


def forward_elimination(a, right):
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("Матрица должна быть непустой и квадратной.")
    if len(right) != n or not right[0] or any(len(row) != len(right[0]) for row in right):
        raise ValueError("Некорректная размерность правой части.")
    augmented = [[float(v) for v in a[i]] + [float(v) for v in right[i]] for i in range(n)]
    if any(not isfinite(v) for row in augmented for v in row):
        raise ValueError("Все элементы должны быть конечными числами.")
    swaps = 0
    pivot_row = 0
    for column in range(n):
        pivot = next((i for i in range(pivot_row, n) if augmented[i][column] != 0), None)
        if pivot is None:
            # Пропуск нулевого столбца позволяет распознать вырожденные системы.
            continue
        if pivot != pivot_row:
            augmented[pivot_row], augmented[pivot] = augmented[pivot], augmented[pivot_row]
            swaps += 1
        for i in range(pivot_row + 1, n):
            multiplier = -augmented[i][column] / augmented[pivot_row][column]
            for j in range(column + 1, len(augmented[i])):
                augmented[i][j] += multiplier * augmented[pivot_row][j]
            augmented[i][column] = 0.0
        pivot_row += 1
    return augmented, swaps


def back_substitution(augmented, rhs_column=None):
    """Обратный ход: x_i = (b_i - сумма a_ij*x_j) / a_ii."""
    n = len(augmented)
    rhs_column = n if rhs_column is None else rhs_column
    for row in augmented:
        if all(value == 0 for value in row[:n]) and row[rhs_column] != 0:
            raise ValueError("Система несовместна: решений нет.")
    if any(augmented[i][i] == 0 for i in range(n)):
        raise ValueError("Система имеет бесконечно много решений.")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (augmented[i][rhs_column] - sum(augmented[i][j] * x[j]
                                                for j in range(i + 1, n))) / augmented[i][i]
    return x


def determinant(upper, swaps):
    result = (-1.0) ** swaps
    for i in range(len(upper)):
        result *= upper[i][i]
    return result


def solve_system(a, b):
    """Решает Ax=b прямым и обратным ходом метода Гаусса."""
    augmented, _ = forward_elimination(a, [[value] for value in b])
    return back_substitution(augmented)


def inverse_matrix(a):
    """Один прямой ход для [A | E], затем n обратных ходов (замечание 4)."""
    n = len(a)
    identity = [[float(i == j) for j in range(n)] for i in range(n)]
    augmented, _ = forward_elimination(a, identity)
    if any(augmented[i][i] == 0 for i in range(n)):
        raise ValueError("Матрица вырождена: обратной матрицы нет.")
    columns = [back_substitution(augmented, n + j) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def read_numbers(prompt, size):
    while True:
        try:
            values = [float(value) for value in input(prompt).split()]
            if len(values) != size or not all(isfinite(value) for value in values):
                raise ValueError
            return values
        except ValueError:
            print(f"Введите ровно {size} конечных чисел через пробел.")


def parse_matrix_from_console(n):
    print("Введите элементы матрицы A построчно через пробел:")
    return [read_numbers(f"Строка {i + 1}: ", n) for i in range(n)]


def parse_vector_from_console(size):
    return read_numbers("b = ", size)


def get_hardcoded_data():
    return ([[-7.0, 3.0, -4.0, 7.0],
             [8.0, -1.0, -7.0, 6.0],
             [9.0, 9.0, 3.0, -6.0],
             [-7.0, -9.0, -8.0, -5.0]],
            [-126.0, 29.0, 27.0, 34.0])


def make_input():
    print("1. Ввести матрицу A и вектор b вручную")
    print("2. Использовать заранее заданные значения")
    while True:
        choice = input("Ваш выбор: ").strip()
        if choice == "2":
            return get_hardcoded_data()
        if choice == "1":
            while True:
                try:
                    n = int(input("Размерность N: "))
                    if n < 1:
                        raise ValueError
                    break
                except ValueError:
                    print("Введите положительное целое число.")
            return parse_matrix_from_console(n), parse_vector_from_console(n)
        print("Неверный выбор. Введите 1 или 2.")


def print_matrix(title, matrix):
    print(title)
    for row in matrix:
        print(" ".join(f"{value:14.8g}" for value in row))


def main():
    try:
        a, b = get_hardcoded_data() if "--example" in sys.argv else make_input()
        n = len(a)
        # [A | b | E]: общий прямой ход для системы и столбцов обратной матрицы.
        right = [[b[i]] + [float(i == j) for j in range(n)] for i in range(n)]
        print_matrix("Исходная расширенная матрица [A | b | E]:", [a[i] + right[i] for i in range(n)])
        augmented, swaps = forward_elimination(a, right)
        print_matrix("После прямого хода [U | b' | E']:", augmented)
        print(f"Число перестановок строк: {swaps}")
        print(f"Определитель A: {determinant(augmented, swaps):.12g}")
        x = back_substitution(augmented)
        print("Решение СЛАУ (обратный ход):")
        for i, value in enumerate(x, 1):
            print(f"x{i} = {value:.12g}")
        columns = [back_substitution(augmented, n + 1 + j) for j in range(n)]
        inverse = [[columns[j][i] for j in range(n)] for i in range(n)]
        print_matrix("Обратная матрица A:", inverse)
        residual = max(abs(sum(a[i][j] * x[j] for j in range(n)) - b[i]) for i in range(n))
        inverse_error = max(abs(sum(a[i][k] * inverse[k][j] for k in range(n)) - (i == j))
                            for i in range(n) for j in range(n))
        print(f"Невязка max|Ax - b|: {residual:.3e}")
        print(f"Погрешность max|AA^(-1) - E|: {inverse_error:.3e}")
    except (ValueError, OverflowError) as error:
        print(f"Ошибка: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
