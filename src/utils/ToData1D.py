import h5py
import numpy as np

# =========================================================================
#                            ПАРМЕТРЫ
# =========================================================================

psi_initial_path = "psi_initial.npy"
final_path = (
    # "/home/denis/Programs/atoms_and_ions/DATA/br/br1e1d_N120_dx05_external.hdf5"
)
dim = 1

# =========================================================================
#             Координатная сетка и начальная волновая функция
# =========================================================================

x0 = np.load("x0.npy").astype(np.float32)
dx = x0[1] - x0[0]
n = len(x0)
psi_initial = np.load("psi_initial.npy").astype(np.complex64)

# =========================================================================
#                    ToData
# =========================================================================
with h5py.File(final_path, "w") as f:
    gXspace = f.create_group("Xspace")
    gXspace.attrs["prs"] = f"N={n}, dx={dx}"
    # for i in range(4):
    gXspace.create_dataset("x0", data=x0)
    gXspace.create_dataset("N", data=[n])
    gXspace.create_dataset("dx", data=[dx])

    gwf = f.create_group("WaveFunction")
    gwf.attrs["comments"] = "волновая функция чистая"
    gwf.create_dataset("psi_im", data=psi_initial.imag)
    gwf.create_dataset("psi_re", data=psi_initial.real)

    gap = f.create_group("AtomicPotential")
    gap.attrs["type"] = "короткодействующий потенциал"
    # gap.attrs["smoothing_coefficients"] = "a->e_core"
    gap.attrs["c1"] = 0.48
    gap.attrs["c2"] = 0.76
    gap.attrs["analitic_expression"] = "-c1*exp[-x^2/c2^2]"
    # gap.create_dataset("atomic_potential", data=V_potential)

    gTimeFFT = f.create_group("TimeFFT")
    gTimeFFT.attrs["Delta_Energy [eV]"] = 0.08548546
    gTimeFFT.attrs["Energy [eV]"] = -3.3339615
    gTimeFFT.attrs["dt_in_TimeFFT"] = 0.2
    gTimeFFT.attrs["n_step_in_TimeFFT"] = 20
    # del gTimeFFT.attrs["nt_in_TimeFFT"]
    gTimeFFT.attrs["nt_in_TimeFFT"] = 500
    gTimeFFT.create_dataset("Energy", data=-3.3339615)
