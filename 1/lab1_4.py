"""Вращения Якоби для симметричных матриц, п. 1.2.2 сборника."""
from math import atan, pi, sin, cos, sqrt
from numerics import validate_matrix, validate_eps, identity, transpose, multiply, input_matrix, residual


def get_example():
    return [[-7, -6, 8], [-6, 3, -7], [8, -7, 4]]


def off_diagonal(a):
    return sqrt(sum(a[i][j] ** 2 for i in range(len(a)) for j in range(i + 1, len(a))))


def jacobi_rotation(a, eps=1e-6, max_iterations=100000):
    n = validate_matrix(a)
    validate_eps(eps)
    if any(a[i][j] != a[j][i] for i in range(n) for j in range(n)):
        raise ValueError("Метод вращений требует симметричную матрицу.")
    current = [[float(v) for v in row] for row in a]
    vectors = identity(n)
    history = [off_diagonal(current)]
    while history[-1] > eps:
        if len(history) - 1 >= max_iterations:
            raise ValueError("Превышено максимальное число вращений.")
        i, j = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda pair: abs(current[pair[0]][pair[1]]))
        angle = pi / 4 if current[i][i] == current[j][j] else 0.5 * atan(
            2 * current[i][j] / (current[i][i] - current[j][j]))
        rotation = identity(n)
        rotation[i][i] = rotation[j][j] = cos(angle)
        rotation[i][j], rotation[j][i] = -sin(angle), sin(angle)
        current = multiply(multiply(transpose(rotation), current), rotation)
        # Подавление асимметрии от округлений и явный нуль выбранного элемента.
        for r in range(n):
            for c in range(r + 1, n):
                current[r][c] = current[c][r] = (current[r][c] + current[c][r]) / 2
        current[i][j] = current[j][i] = 0.0
        vectors = multiply(vectors, rotation)
        history.append(off_diagonal(current))
    return [current[i][i] for i in range(n)], vectors, history


def main(example=False, eps=1e-6):
    a = get_example() if example else input_matrix()
    values, vectors, history = jacobi_rotation(a, eps)
    for j, value in enumerate(values):
        vector = [row[j] for row in vectors]
        print("lambda =", value, "v =", vector, "невязка:", residual(a, vector, [value*x for x in vector]))
    print("Итерация | норма внедиагональной части")
    for step, error in enumerate(history):
        print(step, error)
    return 0
