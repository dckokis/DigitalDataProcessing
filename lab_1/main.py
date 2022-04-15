import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

T = 2


def frequency():
    return 2 * np.pi / T


def coefficients(N, coeff_type_a: bool, function):
    coeffs = np.zeros(N)
    for i in range(N):
        if coeff_type_a:
            coeffs[i] = integrate.quad(lambda x: 2 * function(x) * np.cos(i * x * frequency()) / T, 0, T)[0]
        else:
            coeffs[i] = integrate.quad(lambda x: 2 * function(x) * np.sin(i * x * frequency()) / T, 0, T)[0]
    return coeffs


def give_function_as_array(N, begin, end, step, function):
    res = np.zeros(N)
    for i in range(N):
        if begin <= end:
            begin += step
            res[i] = function(begin)
    return res


def fourier_seq(t, a, b):
    fourier = np.zeros(len(t))
    for i in range(len(fourier)):
        cur = a[0] / 2
        for j in range(1, len(a)): cur += a[j] * np.cos(j * t[i] * frequency())
        for j in range(1, len(b)): cur += b[j] * np.sin(j * t[i] * frequency())
        fourier[i] = cur
    return fourier


def main():
    # meandr = lambda t: 1 if t < T / 2 else -1
    saw = lambda t: 1 - t if t > 0 else -(1 - (-t))
    plt.figure()

    a = coefficients(100, True, saw)
    b = coefficients(100, False, saw)

    plt.subplot(3, 1, 1)
    plt.plot(a, '*')
    plt.plot(b, '+')
    plt.xlabel('n')
    plt.ylabel('coefficients a_n b_n')

    m = give_function_as_array(200, -1, 1, 0.01, saw)
    f = fourier_seq(np.arange(-1, 1, 0.01), np.zeros(1), b)

    plt.subplot(3, 1, 2)
    plt.plot(f, '-')
    plt.plot(m, '--')
    plt.xlabel('time')
    plt.ylabel('signal')

    plt.subplot(3, 1, 3)
    plt.plot(np.abs(m - f))
    plt.xlabel('time')
    plt.ylabel('error')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


main()
