import numpy as np
import matplotlib.pyplot as plt
from Autoencoder import AutoE


#--->>> import your data <<<---#
data = np.genfromtxt("2r4.csv", delimiter=",")
data = data[:, 2:403]
data = np.transpose(data)
data = np.reshape(data, (-1, 50, 150, 1))

# lets see our data
shape = (50, 150, 1)
time_step = 0.01
cap_time = 4


time_vec = np.arange(0, 4+time_step, time_step)

numelems = 6  # number of element to show in picture
numcols = 3
fig, axes = plt.subplots(
    nrows=int(np.ceil(numelems/numcols)), ncols=numcols, figsize=(6, 6))
idx = np.round(np.linspace(0, len(data[:, 0, 0]) - 1, numelems)).astype(int)

for i in range(numelems):

    ax1 = axes.flat[i]
    ax1.set_title("t={:.2f}".format(time_vec[idx[i]]))
    im = ax1.imshow(
        data[idx[i], :, :],
        cmap="jet",
        interpolation="quadric",
        vmin=0,
        vmax=1)
    fig.colorbar(im, ax=ax1, shrink=0.4)

plt.show()

# --->>>>>>>> Animation <<<<<<---
# Let see an animation after completeing
# the simulation make an animation for comparison


#---->>> Train the Auroencoder <<<---#

# Initializing our Neural Network class
bottelneck = 4
filter_shape = (5, 5)
NN = AutoE(shape, bottelneck, filter_shape)
NN.create_model(summary=True)

# The data is normilized

# Lets split data to train and test
test_frac = 0.2
train_ix = np.random.choice(len(data[0, 0, :]), size=int(
    (1-test_frac)*len(data[0, 0, :])), replace=False)
test_ix = np.setdiff1d(np.arange(len(data[0, 0, :])), train_ix)

train_data = data[train_ix, :, :]
test_data = data[test_ix, :, :]


# train the neural network
epoch = 300
EACH_N_EPOCH = 5
lr = 0.0001
batch_size = 8
history = NN.train(train_data, test_data, epoch, EACH_N_EPOCH, batch_size, lr)
NN.autoencoder.save('autoencoder.h5')
np.save('autoencoder_history', history)
