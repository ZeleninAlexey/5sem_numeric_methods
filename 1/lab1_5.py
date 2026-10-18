"""QR-разложение отражениями Хаусхолдера и QR-итерации со сдвигом."""
import cmath
from math import sqrt
from numerics import validate_matrix, validate_eps, identity, multiply, input_matrix, print_matrix


def get_example():
    return [[-1, 4, -4], [2, -5, 0], [-8, -2, 0]]


def reflector(x):
    length = sqrt(sum(abs(v)**2 for v in x))
    if length == 0:
        return None
    phase = x[0] / abs(x[0]) if x[0] != 0 else 1
    v = list(x)
    v[0] += phase * length
    scale = sqrt(sum(abs(z)**2 for z in v))
    return [z / scale for z in v]


def qr_decomposition(a):
    # Комплексная арифметика позволяет использовать комплексные сдвиги.
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("Нужна непустая квадратная матрица.")
    if any(not (cmath.isfinite(v)) for row in a for v in row):
        raise ValueError("Нужны конечные элементы.")
    r = [[complex(v) for v in row] for row in a]
    q = [[complex(v) for v in row] for row in identity(n)]
    for k in range(n):
        v = reflector([r[i][k] for i in range(k, n)])
        if v is None:
            continue
        # R <- H R, Q <- Q H, H = E - 2vv*, H*H = E.
        for j in range(k, n):
            dot = sum(v[t].conjugate() * r[k+t][j] for t in range(len(v)))
            for t in range(len(v)):
                r[k+t][j] -= 2 * v[t] * dot
        for i in range(n):
            dot = sum(q[i][k+t] * v[t] for t in range(len(v)))
            for t in range(len(v)):
                q[i][k+t] -= 2 * dot * v[t].conjugate()
        for i in range(k+1, n):
            r[i][k] = 0j
    return q, r


def hessenberg(a):
    h = [[complex(v) for v in row] for row in a]
    n = len(h)
    for k in range(n-2):
        v = reflector([h[i][k] for i in range(k+1, n)])
        if v is None:
            continue
        for j in range(n):
            dot = sum(v[t].conjugate() * h[k+1+t][j] for t in range(len(v)))
            for t in range(len(v)):
                h[k+1+t][j] -= 2 * v[t] * dot
        for i in range(n):
            dot = sum(h[i][k+1+t] * v[t] for t in range(len(v)))
            for t in range(len(v)):
                h[i][k+1+t] -= 2 * dot * v[t].conjugate()
        for i in range(k+2, n):
            h[i][k] = 0j
    return h


def qr_eigenvalues(a, eps=1e-6, max_iterations=10000):
    validate_matrix(a)
    validate_eps(eps)
    h = hessenberg(a)
    values, history = [], []
    while len(h) > 1:
        n = len(h)
        coupling = abs(h[-1][-2])
        if coupling <= eps * max(1.0, abs(h[-1][-1]) + abs(h[-2][-2])):
            values.append(h[-1][-1])
            h = [row[:-1] for row in h[:-1]]
            continue
        if len(history) >= max_iterations:
            raise ValueError("QR-алгоритм не сошёлся за заданное число итераций.")
        aa, bb, cc, dd = h[-2][-2], h[-2][-1], h[-1][-2], h[-1][-1]
        delta = cmath.sqrt((aa-dd)**2 + 4*bb*cc)
        roots = [(aa+dd+delta)/2, (aa+dd-delta)/2]
        shift = min(roots, key=lambda z: abs(z-dd))
        shifted = [[h[i][j] - (shift if i == j else 0) for j in range(n)] for i in range(n)]
        q, r = qr_decomposition(shifted)
        h = multiply(r, q)
        for i in range(n):
            h[i][i] += shift
            for j in range(i-1):
                h[i][j] = 0j
        history.append((n, abs(h[-1][-2])))
    values.append(h[0][0])
    return values, history


def main(example=False, eps=1e-6):
    a = get_example() if example else input_matrix()
    q, r = qr_decomposition(a)
    print_matrix("Q:", q)
    print_matrix("R:", r)
    values, history = qr_eigenvalues(a, eps)
    print("Собственные значения:")
    for value in values:
        print(value)
    print("Число QR-итераций:", len(history))
    return 0
