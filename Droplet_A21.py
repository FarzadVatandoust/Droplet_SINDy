import trace
import numpy as np
import pysindy as ps
import matplotlib.pylab as plt
from alg21 import alg21
from sklearn.preprocessing import PolynomialFeatures

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
U, S, V, data_mean = Data2SVD(data, data_number, shape)

mod_number = 10

# calculate POD coefficients
u_gt = np.zeros((data_number, mod_number))

for i in range(data_number):
    u_gt[i, :] = np.matmul(np.array([data[:, i]]), U[:, : mod_number])


# We can add noise Here
u_hat =  u_gt 

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
LAMBDA = 0.001

coefficients = np.zeros((len(a[0]), mod_number))

for i in range(mod_number):
    x, _, _ = alg21(a, b[:, i], LAMBDA)
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


# Simulate the flow field with the identified equation
# Here we can choose some point in our data POD_C and u_identified 
# and simulate its field

# We chose 10 even distributed number
numelems = 20
idx = np.round(np.linspace(0, len(u_hat) - 1, numelems)).astype(int)

# We have 3 types of the data
# 1 the original field
field_gt = data[:, idx]
# 2 field that has been created by inversing SVD from original data
field_inv = np.matmul(u_hat[idx], U[:, : mod_number].transpose())
# 3 field that has been created by inversing SVD from predicted data
field_pre = np.matmul(u_identified[idx], U[:, : mod_number].transpose())

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

    im = ax3.imshow(
    field_pre[i, :].reshape(shape),
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