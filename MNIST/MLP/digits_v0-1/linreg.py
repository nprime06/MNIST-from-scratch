import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(42)
n = 100
a, = 2 * np.random.randn(1)
b, = 2 * np.random.randn(1)
X = 2 * np.random.rand(n, 1)
y = a + b * X + np.random.randn(n, 1)

muX = (X.T @ np.ones((n, 1)))/n
muy = (y.T @ np.ones((n, 1)))/n
lsrm = ((X - muX).T @ (y - muy))/((X - muX).T @ (X - muX))
lsrb = muy - lsrm * muX

iterations = 1000
alpha = 0.1

m, b = 1, 0
pred = m * X + b
diff = pred - y

linregx = np.array([[-0.25], [2.25]])
linregy = m * linregx + b
lsry = lsrm * linregx + lsrb

fig, ax = plt.subplots()
sc = ax.scatter(X,y)
line, = ax.plot(linregx, linregy, label = f"y = {m}x + {b}")
lsr, = ax.plot(linregx, lsry, label = f"y = {lsrm}x + {lsrb}", color = "red", linestyle = "--")

label = ax.legend().get_texts()[0]

ax.set(xlim = (-0.25, 2.25), ylim = (a - b - 2, a + b + 2))


for i in range(iterations):
    ax.set_title(f"Iteration: {(i+1)}/{iterations}, learning rate = {alpha}")

    pred = m * X + b
    diff = pred - y
    m -= alpha * (2/n) * (diff.T @ X)
    b -= alpha * (2/n) * (diff.T @ np.ones((n,1)))

    linregy = m * linregx + b
    line.set_ydata(linregy)
    label.set_text(f"y = {m}x + {b}")
    # plt.draw()
    plt.pause(0.015)

plt.show()
