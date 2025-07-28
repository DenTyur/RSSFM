import h5py
import numpy as np

# =========================================================================
#                            ПАРМЕТРЫ
# =========================================================================

final_path = (
    "/home/denis/Programs/atoms_and_ions/DATA/He/He1e3d_dx02_N100_external.hdf5"
)
dim = 1

# =========================================================================
#             Координатная сетка и начальная волновая функция из npy
# =========================================================================

x0 = np.load("x0.npy").astype(np.float32)
x1 = np.load("x1.npy").astype(np.float32)
x2 = np.load("x2.npy").astype(np.float32)
dx0 = x0[1] - x0[0]
dx1 = x1[1] - x1[0]
dx2 = x2[1] - x2[0]
n0 = len(x0)
n1 = len(x1)
n2 = len(x2)
psi_initial = np.load("psi_initial.npy").astype(np.complex64)

# =========================================================================
#             Координатная сетка и начальная волновая функция из hdf5
# =========================================================================

# with h5py.File(psi_initial_path, "r") as f:
#     psi_initial = f["psi_xy_t_last"][:]
#     x0 = f["x"][:]
#
# np.save("psi_initial.npy", psi_initial.astype(np.complex64))
# np.save("x0.npy", x0.astype(np.float32))
# np.save("x1.npy", x0.astype(np.float32))

# =========================================================================
#                    ToData
# =========================================================================
with h5py.File(final_path, "w") as f:
    gXspace = f.create_group("Xspace")
    gXspace.attrs["n"] = [n0, n1, n2]
    gXspace.attrs["dx"] = [dx0, dx1, dx2]
    gXspace.create_dataset("x0", data=x0)
    gXspace.create_dataset("x1", data=x1)
    gXspace.create_dataset("x2", data=x2)
    gXspace.create_dataset("n", data=[n0, n1, n2])
    gXspace.create_dataset("dx", data=[dx0, dx1, dx2])

    gwf = f.create_group("WaveFunction")
    gwf.attrs["comments"] = "волновая функция. слегка грязная на масштабе 1e-8"
    gwf.create_dataset("psi_im", data=psi_initial.imag)
    gwf.create_dataset("psi_re", data=psi_initial.real)

    gap = f.create_group("AtomicPotential")
    gap.attrs["type"] = "сглаженный TongLin"
    gap.attrs["smoothing_coefficient"] = "a->e_core"

    gap.attrs["z"] = 1.0
    gap.attrs["a1"] = 1.375
    gap.attrs["a2"] = 0.662
    gap.attrs["a3"] = -1.325
    gap.attrs["a4"] = 1.236
    gap.attrs["a5"] = -0.231
    gap.attrs["a6"] = 0.480
    gap.attrs["a"] = 0.1
    gap.attrs["r"] = "sqrt(x0^2+x1^2+x2^2+a^2)"
    gap.attrs["analitic_expression"] = (
        "-(z + a1 * exp(-a2 * r) + a3 * r * exp(-a4 * r) + a5 * exp(-a6 * r)) / r"
    )
    # gap.create_dataset("atomic_potential", data=V_potential)

    gTimeFFT = f.create_group("TimeFFT")
    gTimeFFT.attrs["Delta_Energy [eV]"] = 0.17
    gTimeFFT.attrs["Energy [eV]"] = -24.62
    gTimeFFT.attrs["dt_in_TimeFFT"] = 0.01
    gTimeFFT.attrs["n_step_in_TimeFFT"] = 200
    # del gTimeFFT.attrs["nt_in_TimeFFT"]
    gTimeFFT.attrs["nt_in_TimeFFT"] = 500
    gTimeFFT.create_dataset("Energy", data=-24.62)
