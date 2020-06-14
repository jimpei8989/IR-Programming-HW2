import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(2 - x)

def g(x):
    return -np.log(x)

x = np.arange(0, 1, 0.001)[1:]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(x, f(x), color='#5cbaf7')
ax[0].set_title('Original')
ax[1].plot(x, g(x), color='#5bd28b')
ax[1].set_title('BCE-like')

fig.savefig('Prob3.png')

