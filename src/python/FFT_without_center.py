import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import time
import os
import gc
import pyfftw

basedir = os.path.abspath(os.getcwd())
src_dir = os.path.abspath(os.path.join(basedir, ".."))

x = np.load(src_dir + "/arrays_saved/x0.npy")
y = np.load(src_dir + "/arrays_saved/x1.npy")
px = np.load(src_dir + "/arrays_saved/p0.npy")
py = np.load(src_dir + "/arrays_saved/p1.npy")
t = np.load(src_dir + "/arrays_saved/time_evol/t.npy")


def get_step(x):
    return x[1] - x[0]


dx = get_step(x)
dy = get_step(y)
dpx = get_step(px)
dpy = get_step(py)

Nx = len(x)
Ny = len(y)
Npx = len(px)
Npy = len(py)

NxMesh, NyMesh = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

X, Y = np.meshgrid(x, y, indexing="ij")
PX, PY = np.meshgrid(px, py, indexing="ij")


def plotter(psi, X, Y, title, save_path, vmin):
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, 8), layout="constrained")
    axs.set(
        aspect="equal",
        title=title,
    )
    b = axs.pcolormesh(
        X,
        Y,
        np.abs(psi) ** 2,
        cmap=cm.jet,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=1),
    )
    cb = plt.colorbar(b, ax=axs)
    fig.savefig(save_path)
    axs.clear()
    cb.remove()
    gc.collect()


def get_psi_mod(psi, px, py, x, y):
    return (
        psi_x
        * np.exp(-1j * px[0] * x)
        * dx
        / np.sqrt(2 * np.pi)
        * np.exp(-1j * py[0] * y)
        * dy
        / np.sqrt(2 * np.pi)
    )


def centering_FFT(psi_p, Npx, Npy):
    psi_p_centered = np.zeros_like(psi_p)
    psi_p_centered[: Npx // 2, : Npy // 2] = psi_p[Npx // 2 :, Npy // 2 :]
    psi_p_centered[Npx // 2 :, : Npy // 2] = psi_p[: Npx // 2, Npy // 2 :]
    psi_p_centered[Npx // 2 :, Npy // 2 :] = psi_p[: Npx // 2, : Npy // 2]
    psi_p_centered[: Npx // 2, Npy // 2 :] = psi_p[Npx // 2 :, : Npy // 2]
    return psi_p_centered


def demodify_psi_p(psi_p_centered, x, y, dx, dy, NxMesh, NyMesh):
    return (
        psi_p_centered
        * np.exp(-1j * x[0] * dx * NxMesh)
        * np.exp(-1j * y[0] * dy * NyMesh)
    )


i_step = 21
r_cut = 20
procs = 4
vmin_x = 1e-8

ind_max = np.abs(x - r_cut).argmin()
ind_min = np.abs(x + r_cut).argmin()


# планируем FFT
a = pyfftw.empty_aligned((Nx, Ny), dtype="complex64")
b = pyfftw.empty_aligned((Nx, Ny), dtype="complex64")
c = pyfftw.empty_aligned((Nx, Ny), dtype="complex64")
fft_object = pyfftw.FFTW(a, b, axes=(0, 1), threads=procs)
ifft_object = pyfftw.FFTW(
    b,
    c,
    axes=(0, 1),
    threads=procs,
    direction="FFTW_FORWARD",
)

# без вырезания середины
psi_x = np.load(src_dir + f"/arrays_saved/time_evol/psi_x/psi_t_{i_step}.npy")

plotter(
    psi_x,
    X,
    Y,
    f"external |psi(x, y)|^2 step={i_step} of {len(t)}; t = {t[i_step]:.{5}f} a.u.",
    src_dir + f"/imgs/time_evol/psi_x_cutted/psi_x_{i_step}.png",
    vmin_x,
)
np.save(src_dir + f"/arrays_saved/time_evol/psi_x_cutted/psi_x_{i_step}.npy", psi_x)


psi_x_mod = get_psi_mod(psi_x, px, py, x, y)

a[:] = psi_x_mod
psi_p = fft_object()
psi_p_centered = centering_FFT(psi_p, Npx, Npy)
psi_p_demodified = demodify_psi_p(psi_p_centered, x, y, dx, dy, NxMesh, NyMesh)

plotter(
    psi_p_demodified,
    PX,
    PY,
    f"external |psi(px, py)|^2 step={i_step} of {len(t)}; t = {t[i_step]:.{5}f} a.u.",
    src_dir + f"/imgs/time_evol/psi_p_cutted/psi_p_{i_step}.png",
    1e-5,
)
np.save(src_dir + f"/arrays_saved/time_evol/psi_p_cutted/psi_p_{i_step}.npy", psi_p)

# с вырезанием середины
psi_x[X**2 + Y**2 < r_cut**2] = vmin_x

plotter(
    psi_x,
    X,
    Y,
    f"external |psi(x, y)|^2 step={i_step} of {len(t)}; t = {t[i_step]:.{5}f} a.u.",
    src_dir + f"/imgs/time_evol/psi_x_cutted/psi_x_cutted{i_step}.png",
    vmin_x,
)
np.save(
    src_dir + f"/arrays_saved/time_evol/psi_x_cutted/psi_x_cutted_{i_step}.png", psi_x
)

psi_x_mod = get_psi_mod(psi_x, px, py, x, y)
a[:] = psi_x_mod
psi_p = fft_object()
psi_p_centered = centering_FFT(psi_p, Npx, Npy)
psi_p_demodified = demodify_psi_p(psi_p_centered, x, y, dx, dy, NxMesh, NyMesh)

plotter(
    psi_p_demodified,
    PX,
    PY,
    f"external |psi(px, py)|^2 step={i_step} of {len(t)}; t = {t[i_step]:.{5}f} a.u.",
    src_dir + f"/imgs/time_evol/psi_p_cutted/psi_p_cutted_{i_step}.png",
    1e-5,
)
np.save(
    src_dir + f"/arrays_saved/time_evol/psi_p_cutted/psi_p_cutted_{i_step}.npy",
    psi_p_demodified,
)
