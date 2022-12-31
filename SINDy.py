import numpy as np
from Autoencoder import AutoE

import matplotlib.pylab as plt
import matplotlib.animation as animation

from alg21 import alg21
from sklearn.preprocessing import PolynomialFeatures

import ffmpeg
#--->>> import your data <<<---#
data = np.genfromtxt("2r4.csv", delimiter=",")
data = data[:, 2:403]
data = np.transpose(data)
data = np.reshape(data, (-1, 50 , 150, 1))

# lets see our data
shape = (50, 150, 1)
time_step = 0.01
cap_time = 4

#--->>> Load the Autoencoder <<<---#
NN = AutoE(shape)
NN.create_model(summary=False)
NN.autoencoder.load_weights('autoencoder.h5')


#--->>> Create lower dimsensional data <<<---#
l_data = NN.encoder(data).numpy()




#--->>> Apply the SINDy Algo <<<---#
mod_number = 3
dt = 0.01 # Time step

u_hat =  l_data

# Crearing the a and b Marix in ax=b
#a_names = poly.get_feature_names_out(['u1','u2','u3'])
# p1 = 3
poly = PolynomialFeatures(degree=5, interaction_only=False)
p1 = poly.fit_transform(u_hat)

# p2=p3=1
poly = PolynomialFeatures(degree=3, interaction_only=False)
p2 = np.delete(np.sin(poly.fit_transform(u_hat)), 0, 1)
p3 = np.delete(np.cos(poly.fit_transform(u_hat)), 0, 1)

a = np.hstack((p1, p2, p3))

# Compute the derivative from data 
b = np.zeros((len(u_hat), len(u_hat[0])))

for j in range(len(u_hat[0])):
    b[0, j] = (u_hat[1, j] - u_hat[0, j])/dt
    for i in range(1, len(u_hat)-1):
        b[i, j] = (u_hat[i+1, j] - u_hat[i-1, j]) / (2 * dt)
    b[-1, j] = (u_hat[-1, j] - u_hat[-2, j]) / dt



# Apply algo21
LAMBDA = 0.01

coefficients = np.zeros((len(a[0]), mod_number))

for i in range(mod_number):
    x, _, _ = alg21(a, b[:, i], LAMBDA)

    # temporary fix for column b of zero
    if len(x[0]) == 0:
        x =  np.zeros((len(x), 1))

    coefficients[:, i] = x[:, -1]

u_dot_identified = np.matmul(a, coefficients)
u_identified = np.zeros((len(u_hat), mod_number))
for i in range(mod_number):
    u_identified[0, i] = u_hat[0, i]


for i in range(mod_number):
    for j in range(1, len(u_hat)):
        u_identified[j, i] = u_dot_identified[j-1, i] * \
            dt + u_identified[j-1, i]


fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.set_title("Lorenz System")
plt.plot(u_hat[:, 0], u_hat[:, 1], u_hat[:, 2])
ax1.set(xlabel="$u1$", ylabel="$u2$", zlabel="$u3$")


ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.set_title("Lorenz System with Noise")
plt.plot(u_hat[:, 0], u_hat[:, 1], u_hat[:, 2])
ax2.set(xlabel="$u1$", ylabel="$u2$", zlabel="$u3$")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.set_title("Identified System")
plt.plot(u_identified[:, 0], u_identified[:, 1], u_identified[:, 2])
ax3.set(xlabel="$u1$", ylabel="$u2$", zlabel="$u3$")

plt.show()

#--->>> Simulate the result <<<---#

# We chose 10 even distributed number
numelems = 20
idx = np.round(np.linspace(0, len(u_hat) - 1, numelems)).astype(int)

# We have 3 types of the data
# 1 the original field
field_gt = data[idx, :]
# 2 field that has been created by just inversing
# the data dimensionality rducing model
field_inv = NN.decoder(u_hat[idx, :]).numpy()

# 3 field that has been generated from inversing the predicted data 
field_pre = NN.decoder(u_identified[idx, :]).numpy()

# ploting the data

fig, axes = plt.subplots(nrows=numelems, ncols=3)
axes.flat[0].set_title("Full Simulation")
axes.flat[1].set_title("Inverse Simulation")
axes.flat[2].set_title("Identified System")

for i in range(numelems):
    ax1, ax2, ax3 = axes.flat[3 * i], axes.flat[3 * i + 1], axes.flat[3 * i + 2]
    
    plt.annotate(
    "idx=%.3f " % idx[i],
    xy=(0.03, ax1.get_position().y0 + ax1.get_position().height / 2),
    xycoords="figure fraction",)

    im = ax1.imshow(
    field_gt[i, :],
    cmap="jet",
    interpolation="quadric",
    vmin=0,
    vmax=1)

    im = ax2.imshow(
    field_inv[i, :],
    cmap="jet",
    interpolation="quadric",
    vmin=0,
    vmax=1)

    im = ax3.imshow(
    field_pre[i, :],
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

# animation
field_gt = data
field_inv = NN.decoder(u_hat).numpy()
field_pre = NN.decoder(u_identified).numpy()

fig, axes = plt.subplots(nrows=3, ncols=1)

plt.subplots_adjust(bottom=0.1,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.6)
ax1, ax2, ax3 = axes.flat[0], axes.flat[1], axes.flat[2]
ax1.title.set_text("Full Simulation")
ax2.title.set_text("Inverse Simulation")
ax3.title.set_text("Identified System")

def im_func(i):
    
    im1 = []
    im = ax1.imshow(
    field_gt[i, :],
    cmap="jet",
    interpolation="quadric",
    vmin=0,
    vmax=1,
    animated=True)

    im = ax2.imshow(
    field_inv[i, :],
    cmap="jet",
    interpolation="quadric",
    vmin=0,
    vmax=1,
    animated=True)
    

    im = ax3.imshow(
    field_pre[i, :],
    cmap="jet",
    interpolation="quadric",
    vmin=0,
    vmax=1,
    animated=True)


ani = animation.FuncAnimation(fig, im_func)
ani.save("ani")

plt.show()

