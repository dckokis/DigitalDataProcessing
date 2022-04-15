import numpy as np
import matplotlib.pyplot as plt

precision = 0.1
a = -10
b = 10
interval = np.arange(a, b, precision)

n = 2
wc = 1
w1 = 2
w2 = 30
ws1 = 3 * w1
ws2 = 6 * w1
plt.rcParams['figure.dpi'] = 300


def H(w):
    p = 1
    for k in range(n // 2):
        p *= -((w / wc) ** 2 - 2 * 1j * np.sin(np.pi * (2 * k + 1) / (2 * n)) * (w / wc) - 1)
    Hw = 1 / p
    if n % 2 != 0:
        Hw = Hw / (1j * w - 1)
    return Hw


def Hhighh(w):
    p = 1
    for k in range(n // 2):
        p *= -((1 / w) ** 2 - 2 * 1j * np.sin(np.pi * (2 * k + 1) / (2 * n)) * (1 / w) - 1)
    Hw = 1 / p
    if n % 2 != 0:
        Hw = Hw / (1j * (w / wc) - 1)
    return Hw


def butterH():
    return np.vectorize(H)(interval)


def butterH2():
    return np.vectorize(lambda w: 1 / ((w / wc) ** 2 + np.sqrt(2) * (w / wc) + 1))(interval)


def genArr(func):
    return np.vectorize(func)(interval)


def applyButterworthFilter(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * butterH()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def applyButterworthFilterH2(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * butterH2()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def butterFHR():
    return np.vectorize(Hhighh)(interval)


def butterFHR2():
    return np.vectorize(lambda w: 1 / ((wc / w) ** 2 + np.sqrt(2) * (wc / w) + 1))(interval)


def applyButterworthFilterFHR(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * butterFHR()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def applyButterworthFilterFHR2(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * butterFHR2()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def ButterworthArrLow():
    return np.vectorize(lambda w: 1 / (1 + (w / wc) ** (2 * n)))(interval)


def ButterworthArrHigh():
    return np.vectorize(lambda w: 1 / (1 + (wc / w) ** (2 * n)))(interval)


def Hstripe(w):
    p = 1
    for k in range(n // 2):
        p *= -(((w ** 2 + ws1*ws2) / ((ws1 - ws2) * w)) ** 2 - 2 * 1j * np.sin(np.pi * (2 * k + 1) / (2 * n)) * (
                (w ** 2 + ws1*ws2) / ((ws1 - ws2) * w)) - 1)
    Hw = 1 / p
    if n % 2 != 0:
        Hw = Hw / (1j * ((w ** 2 + ws1*ws2) / ((ws1 - ws2) * w)) - 1)
    return Hw


def stripe():
    return np.vectorize(Hstripe)(interval)


def Hreject(w):
    p = 1
    for k in range(n // 2):
        p *= -((1 / ((w ** 2 + ws1*ws2) / ((ws1 - ws2) * w))) ** 2 - 2 * 1j * np.sin(np.pi * (2 * k + 1) / (2 * n)) * (
            (1 / (w ** 2 + ws1*ws2) / ((ws1 - ws2) * w))) - 1)
    Hw = 1 / p
    if n % 2 != 0:
        Hw = Hw / (1j / ((w ** 2 + ws1*ws2) / ((ws1 - ws2) * w)) - 1)
    return Hw


def reject():
    return np.vectorize(Hreject)(interval)


def applyStripeFilter(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * stripe()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def applyRejectFilter(arr):
    tmp = np.split(arr, 2)
    fs = np.concatenate([tmp[1], tmp[0]])
    fs = fs * reject()
    tmp = np.split(fs, 2)
    return np.fft.ifft(np.concatenate([tmp[1], tmp[0]]))


def filtrate(f):
    filtered1 = applyButterworthFilter(np.fft.fft(genArr(f)))
    filtered2 = applyButterworthFilterH2(np.fft.fft(genArr(f)))
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(interval, ButterworthArrLow())
    plt.title("Butterworth")
    plt.subplot(5, 1, 2)
    plt.plot(interval, genArr(f))
    plt.title("Сигнал")
    plt.subplot(5, 1, 3)
    plt.plot(interval, np.abs(np.fft.fft(genArr(f))))
    plt.title("Спектр сигнала")
    plt.subplot(5, 1, 4)
    plt.plot(interval, np.abs(np.fft.fft(applyButterworthFilter(np.fft.fft(genArr(f))))))
    plt.plot(interval, np.abs(np.fft.fft(applyButterworthFilterH2(np.fft.fft(genArr(f))))))
    plt.title("Спектр сигнала после фильтра")
    plt.subplot(5, 1, 5)
    plt.plot(interval, filtered1)
    plt.plot(interval, filtered2)
    plt.title("Сигнал после фильтра")

    plt.subplots_adjust(hspace=1.1)
    plt.show()


def filtrateFHR(f):
    filteredFHR = applyButterworthFilterFHR(np.fft.fft(genArr(f)))
    filteredFHR2 = applyButterworthFilterFHR2(np.fft.fft(genArr(f)))
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(interval, ButterworthArrHigh())
    plt.title("Butterworth")
    plt.subplot(5, 1, 2)
    plt.plot(interval, genArr(f))
    plt.title("Сигнал")
    plt.subplot(5, 1, 3)
    plt.plot(interval, np.abs(np.fft.fft(genArr(f))))
    plt.title("Спектр сигнала")
    plt.subplot(5, 1, 4)
    plt.plot(interval, np.abs(np.fft.fft(applyButterworthFilterFHR(np.fft.fft(genArr(f))))))
    plt.plot(interval, np.abs(np.fft.fft(applyButterworthFilterFHR2(np.fft.fft(genArr(f))))))
    plt.title("Спектр сигнала после ФВЧ")
    plt.subplot(5, 1, 5)
    plt.plot(interval, filteredFHR)
    plt.plot(interval, filteredFHR2)
    plt.title("Сигнал после фильтра ФВЧ")

    plt.subplots_adjust(hspace=1.1)
    plt.show()


def filtrateStripe(f):
    filteredStripe = applyStripeFilter(np.fft.fft(genArr(f)))
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(interval, ButterworthArrLow())
    plt.title("Butterworth")
    plt.subplot(5, 1, 2)
    plt.plot(interval, genArr(f))
    plt.title("Сигнал")
    plt.subplot(5, 1, 3)
    plt.plot(interval, np.abs(np.fft.fft(genArr(f))))
    plt.title("Спектр сигнала")
    plt.subplot(5, 1, 4)
    plt.plot(interval, np.abs(np.fft.fft(applyStripeFilter(np.fft.fft(genArr(f))))))
    plt.title("Спектр сигнала после полосового фильтра")
    plt.subplot(5, 1, 5)
    plt.plot(interval, filteredStripe)
    plt.title("Сигнал после фильтра")

    plt.subplots_adjust(hspace=1.1)
    plt.show()


def filtrateReject(f):
    filteredReject = applyRejectFilter(np.fft.fft(genArr(f)))
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(interval, ButterworthArrLow())
    plt.title("Butterworth")
    plt.subplot(5, 1, 2)
    plt.plot(interval, genArr(f))
    plt.title("Сигнал")
    plt.subplot(5, 1, 3)
    plt.plot(interval, np.abs(np.fft.fft(genArr(f))))
    plt.title("Спектр сигнала")
    plt.subplot(5, 1, 4)
    plt.plot(interval, np.abs(np.fft.fft(applyRejectFilter(np.fft.fft(genArr(f))))))
    plt.title("Спектр сигнала после заграждающего фильтра")
    plt.subplot(5, 1, 5)
    plt.plot(interval, filteredReject)
    plt.title("Сигнал после фильтра")

    plt.subplots_adjust(hspace=1.1)
    plt.show()


def harmonic(a, b):
    return lambda x: a * np.cos(x * w1) + b * np.cos(x * w2)


# filtrate(harmonic(1, 1))
# filtrateFHR(harmonic(1, 1))
filtrateStripe(harmonic(2, 1))
# filtrateReject(harmonic(2, 1))
