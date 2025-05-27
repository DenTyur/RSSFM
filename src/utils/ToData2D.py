import h5py
import numpy as np

# =========================================================================
#                            ПАРМЕТРЫ
# =========================================================================

# psi_initial_path = "br_2e1D_bound_interaction_clean_128_05.hdf5"
# final_path = "/home/denis/Programs/atoms_and_ions/DATA/models/"
dim = 1

# =========================================================================
#             Координатная сетка и начальная волновая функция из npy
# =========================================================================

x0 = np.load("x0.npy").astype(np.float32)
x1 = np.load("x1.npy").astype(np.float32)
dx0 = x0[1] - x0[0]
dx1 = x1[1] - x1[0]
n0 = len(x0)
n1 = len(x1)
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
    gXspace.attrs["n"] = [n0, n1]
    gXspace.attrs["dx"] = [dx0, dx1]
    gXspace.create_dataset("x0", data=x0)
    gXspace.create_dataset("x1", data=x1)
    gXspace.create_dataset("n", data=[n0, n1])
    gXspace.create_dataset("dx", data=[dx0, dx1])

    gwf = f.create_group("WaveFunction")
    gwf.attrs["comments"] = "волновая функция чистая"
    gwf.create_dataset("psi_im", data=psi_initial.imag)
    gwf.create_dataset("psi_re", data=psi_initial.real)

    gap = f.create_group("AtomicPotential")
    gap.attrs["type"] = "сглаженный кулон вместе с е-е взаимодействием"
    gap.attrs["smoothing_coefficients"] = "a->e_core, b->e_e"
    gap.attrs["a"] = 3.15
    gap.attrs["b"] = 7.43
    gap.attrs["analitic_expression"] = (
        "-1/sqrt(x0^2+a^2)-1/sqrt(x1^2+a^2)+1/sqrt((x1-x0)^2+b^2)"
    )
    # gap.create_dataset("atomic_potential", data=V_potential)

    gTimeFFT = f.create_group("TimeFFT")
    gTimeFFT.attrs["Delta_Energy [eV]"] = 0.014247894
    gTimeFFT.attrs["Energy [eV]"] = -10.115864
    gTimeFFT.attrs["dt_in_TimeFFT"] = 0.2
    gTimeFFT.attrs["n_step_in_TimeFFT"] = 10
    # del gTimeFFT.attrs["nt_in_TimeFFT"]
    gTimeFFT.attrs["nt_in_TimeFFT"] = 6000
    gTimeFFT.create_dataset("Energy", data=-10.115864)
