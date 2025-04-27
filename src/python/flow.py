import numpy as np
import matplotlib.pyplot as plt

root = "/home/denis/Programs/atoms_and_ions/br/4D_br_circular/E0035_w004/noninteract_big_grid/external2D/RSSFM2D/src/arrays_saved"
t = np.load(root + "/time_evol/t.npy")
x = np.load(root + "/x0.npy")
dx = x[1] - x[0]
R = 25
min = np.argmin(np.abs(x + R))
max = np.argmin(np.abs(x - R))
print("x border = ", x[min], x[max])
flux = np.zeros_like(t)
prob = np.zeros_like(t)

for i in range(len(t)):
    psi = np.load(root + f"/time_evol/psi_x/psi_t_{i}.npy")
    prob[i] = np.sum(np.abs(psi[min:max, min:max]) ** 2 * dx * dx, axis=(0, 1))

    # left flux: x=x_min, y=y_min..y_max, n = -1
    left_flux = (
        np.sum(
            psi[min, min:max].conj()
            * (psi[min + 1, min:max] - psi[min - 1, min:max])
            / (2 * dx)
        ).imag
        * dx
    )

    # right flux: x=x_max, y=y_min..y_max, n = +1
    right_flux = (
        np.sum(
            psi[max, min:max].conj()
            * (psi[max + 1, min:max] - psi[max - 1, min:max])
            / (2 * dx)
        ).imag
        * dx
    )

    # bottom flux: x=x_min..x_max, y=y_min, n = -1
    bottom_flux = (
        np.sum(
            psi[min:max, min].conj()
            * (psi[min:max, min + 1] - psi[min:max, min - 1])
            / (2 * dx)
        ).imag
        * dx
    )

    # top flux: x=x_min..x_max, y=y_max, n = +1
    top_flux = (
        np.sum(
            psi[min:max, max].conj()
            * (psi[min:max, max + 1] - psi[min:max, max - 1])
            / (2 * dx)
        ).imag
        * dx
    )

    flux[i] = -left_flux + right_flux - bottom_flux + top_flux

tot_flux = flux.sum() * (t[1] - t[0])
final_prob = prob[-1]
print("flux = ", tot_flux)
print("prob = ", final_prob)
print("prob+flux = ", final_prob + tot_flux)

plt.plot(t, flux, "-b")
plt.plot(t, prob, "-r")
plt.show()
