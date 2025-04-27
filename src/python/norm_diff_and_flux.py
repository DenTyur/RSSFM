import numpy as np
import matplotlib.pyplot as plt

t = np.load("../arrays_saved/time_evol/t.npy")
x = np.load("../arrays_saved/x0.npy")
dx = x[1] - x[0]
dt = t[1] - t[0]
flow = np.load("../arrays_saved/time_evol/flow/flow.npy")

dprob_dt = np.zeros(len(t))

for i in range(1, len(t) - 2):
    psi_t0 = np.load(f"../arrays_saved/time_evol/psi_x/psi_x_t_{i}.npy")
    psi_t1 = np.load(f"../arrays_saved/time_evol/psi_x/psi_x_t_{i+2}.npy")

    dprob_dt[i + 1] = (
        (np.abs(psi_t1) ** 2).sum(axis=(0, 1)) * dx * dx
        - (np.abs(psi_t0) ** 2).sum(axis=(0, 1)) * dx * dx
    ) / (2 * dt)
print("tot flow = ", flow.sum() * dt)
print("tot prob = ", np.abs(dprob_dt).sum() * dt)
plt.plot(t, flow.real, color="green", label="flow.real")
plt.plot(t, flow.imag, color="blue", label="flow.imag")
plt.plot(t, np.abs(dprob_dt), color="red", linestyle="--", label="dprob_dt")
plt.legend()
plt.show()
