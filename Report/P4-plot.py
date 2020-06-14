import numpy as np
import matplotlib.pyplot as plt

latentDims = [16, 32, 64, 128, 256]
BCE = [0.03645, 0.03943, 0.03932, 0.03991, 0.04023]
BPR = [0.04828, 0.05430, 0.05692, 0.05536, 0.05397]

x = np.arange(len(latentDims))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(x, BCE, color='#4fa382', label='BCE', marker='^')
ax[0].set_title('BCE')
ax[0].set_xlabel('Latent Dimension')
ax[0].set_ylabel('Public MAP')
ax[0].set_xticks(x)
ax[0].set_xticklabels(map(str, latentDims))

ax[1].plot(x, BPR, color='#6695c2', label='BPR', marker='^')
ax[1].set_title('BPR')
ax[1].set_xlabel('Latent Dimension')
ax[1].set_ylabel('Public MAP')
ax[1].set_xticks(x)
ax[1].set_xticklabels(map(str, latentDims))

fig.savefig('Prob4.png')

