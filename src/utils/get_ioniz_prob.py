import numpy as np
import h5py

file_name = "ioniz_prob.hdf5"
time_instance = float(input("t[a.u.] = "))
with h5py.File(file_name, "r") as f:
    x_surf = f["x_surf"][:]
    t = f["t"][:]

    time_ind = np.argmin(np.abs(t - time_instance))
    for i in range(len(x_surf)):
        ioniz_prob = f[f"ioniz_prob_{i}"][:]
        print(f"x_surf = {x_surf[i]}, W = {ioniz_prob[time_ind]}")
