import numpy as np
import matplotlib.pyplot as plt
from math import sin


def explicit_transcalency_scheme(length, spaceStep, sigma, t, f, f1, f2):
    spaseSteps = int(length / spaceStep)
    dt = spaceStep ** 2 / (2 * sigma)
    timeSteps = int(t / dt)
    solution = np.zeros((timeSteps, spaseSteps))

    for i in range(spaseSteps):
        solution[0][i] = f(i * spaceStep)
    for t in range(timeSteps - 1):
        for x in range(1, spaseSteps - 1):
            solution[t + 1][x] = solution[t][x] + sigma * dt / (spaceStep ** 2) * \
                                 (solution[t][x + 1] - 2 * solution[t][x] + solution[t][x - 1])
            solution[t + 1][0] = f1((t + 1) * dt)
            solution[t + 1][spaseSteps - 1] = f2((t + 1) * dt)
    return solution


def implicit_transcalency_scheme(length, spaceStep, sigma, t, dt, f, f1, f2):
    spaseSteps = int(length / spaceStep) - 1
    timeSteps = int(t / dt) - 1
    solution = np.zeros((timeSteps + 1, spaseSteps + 1))
    for i in range(spaseSteps + 1):
        solution[0][i] = f(i * spaceStep)
    alpha = np.zeros(spaseSteps)
    beta = np.zeros(spaseSteps)
    for ti in range(timeSteps):
        alpha[0] = 0
        beta[0] = f1(dt * (ti + 1))
        for x in range(1, spaseSteps):
            a = -sigma * dt / (spaceStep ** 2)
            b = 1 + 2 * sigma * dt / (spaceStep ** 2)
            c = a
            alpha[x] = -a / (b + c * alpha[x - 1])
            beta[x] = (solution[ti][x] - c * beta[x - 1]) / (b + c * alpha[x - 1])
        solution[ti + 1][spaseSteps] = f2(dt * (ti + 1))
        for x in range(spaseSteps, 0, -1):
            solution[ti + 1][x - 1] = alpha[x - 1] * solution[ti + 1][x] + beta[x - 1]
    return solution


f = lambda x: 10*x
f1 = lambda x: 10
f2 = lambda x: 1000
length = 10
spaceStep = 0.1
sigma = 4
t = 5
dt = 0.01/7.8
sol = explicit_transcalency_scheme(length=length, spaceStep=spaceStep, sigma=sigma, t=t, f=f, f1=f1, f2=f2)
# sol = implicit_transcalency_scheme(length=length, spaceStep=spaceStep, sigma=sigma, t=t, dt=dt, f=f, f1=f1, f2=f2)
plt.pcolormesh(sol)
plt.colorbar()
plt.xlabel('coordinate')
plt.ylabel('time')
plt.show()


