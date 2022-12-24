import trace
import numpy as np
import pysindy as ps
import matplotlib.pylab as plt
from time import sleep
from tqdm import tqdm

# Data to SVD
def Data2SVD(data, data_number, shape):
    # subtracting mean value
    data_mean = data.mean(axis=1).reshape(shape)
    x = data - np.outer(data_mean, np.ones(data_number))

    # calculate SVD
    U, S, V = np.linalg.svd(x, full_matrices=False)

    return U, S, V, data_mean


data = np.genfromtxt("2r4.csv", delimiter=",")
# Every column represent the simulation for different time 
# which is flattend from rectangular domain
data = data[:, 2:203]
# This is the data shape of data on rectangular domain
shape = (50, 150)
dt = 0.01 # Time step
T = 2 # Total time
data_number = data.shape[1]


# Compute SVD
mod_number = 3
PLOT = True
U, S, V, data_mean = Data2SVD(data, data_number, shape)


# calculate POD coefficients
u_gt = np.zeros((data_number, mod_number))

for i in range(data_number):
    u_gt[i, :] = np.matmul(np.array([data[:, i]]), U[:, : mod_number])

numelems = 20
idx = np.round(np.linspace(0, len(u_gt) - 1, numelems)).astype(int)

field_gt = data[:, idx]
# 2 field that has been created by inversing SVD from original data
field_inv = np.matmul(u_gt[idx], U[:, : mod_number].transpose())

#error
mse = 0 
for i in range(numelems):
    dummy = (np.square(field_inv[0, :] - field_gt[:, 0])).mean(axis=0)
    mse = mse + dummy



# ploting the data
if PLOT:
    ncols = 2 
    fig, axes = plt.subplots(nrows=numelems, ncols=ncols)
    axes.flat[0].set_title("Original Data")
    axes.flat[1].set_title("Data from SVD")


    for i in range(numelems):
        ax1, ax2= axes.flat[ncols * i], axes.flat[ncols * i + 1]
        
        plt.annotate(
        "idx=%.3f " % idx[i],
        xy=(0.03, ax1.get_position().y0 + ax1.get_position().height / 2),
        xycoords="figure fraction",)

        im = ax1.imshow(
        field_gt[:, i].reshape(shape),
        cmap="jet",
        interpolation="quadric",
        vmin=0,
        vmax=1)

        im = ax2.imshow(
        field_inv[i, :].reshape(shape),
        cmap="jet",
        interpolation="quadric",
        vmin=0,
        vmax=1)

    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(left=0.2)
    yc = axes.flat[-1].get_position().y0
    hc = axes.flat[0].get_position().y0 + axes.flat[0].get_position().height - yc
    cbar_ax = fig.add_axes([0.85, yc, 0.025, hc])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

