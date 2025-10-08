import numpy as np
import h5py

slice_step = 4
x = [np.load(f"x{i}.npy")[::slice_step] for i in range(4)]
dx = [x[1]-x[0] for x in x]
p0 = [-np.pi/dx for dx in dx]
dp = [2*np.pi/(len(x[i])*dx[i]) for i in range(4)]
p = [np.linspace(p0[i], p0[i]+dp[i]*(len(x[i]-1)), len(x[i]))
     for i in range(4)]
p = [p[::4] for p in p]


for i in range(15):
    print("i=", i)
    psi = np.load(f"psi_x/psi_x_t_{i}.npy")
    with h5py.File(f"psi_x/psi_x_{i}.hdf5", 'w') as f:
        gXspace = f.create_group("Xspace")
        gXspace.create_dataset("x0", data=x[0])
        gXspace.create_dataset("x1", data=x[1])
        gXspace.create_dataset("x2", data=x[2])
        gXspace.create_dataset("x3", data=x[1])

        gPspace = f.create_group("Pspace")
        gPspace.create_dataset("p0", data=p[0])
        gPspace.create_dataset("p1", data=p[1])
        gPspace.create_dataset("p2", data=p[2])
        gPspace.create_dataset("p3", data=p[1])

        gwf = f.create_group("WaveFunction")
        gwf.create_dataset("psi_im", data=psi.imag)
        gwf.create_dataset("psi_re", data=psi.real)
        gwf.attrs["representation"] = "position"
