use rssfm::common::particle::Particle;
use rssfm::common::tspace::Tspace;
use rssfm::dim4::{space::Xspace4D, wave_function::WaveFunction4D};
use rssfm::potentials::{absorbing_potentials::absorbing_potential_4d, potentials};
use rssfm::traits::wave_function::WaveFunction;
use rssfm::F;

fn main() {
    // let out_prefix = ".";
    // let psi_path = "psi_initial.hdf5";

    let mut t = Tspace::new(0., 0.01, 1, 10000);

    let x = Xspace4D::new(
        [-15.0, -15.0, -15.0, -15.0],
        [0.5, 0.5, 0.5, 0.5],
        [64, 64, 64, 64],
    );
    // let mut psi = WaveFunction4D::init_oscillator_4d(x);
    let mut psi = WaveFunction4D::init_from_hdf5(
        "/home/denis/Programs/atoms_and_ions/DATA/br/br2e2d_N64_dx05_interact.hdf5",
    );
    psi.extend(&x);

    let abs_pot = |x: [F; 4]| absorbing_potential_4d(x, 50.0, 0.4);
    let atomic_pot = |x: [F; 4]| potentials::br_2e2d(x);

    let electron = Particle {
        dim: 2,
        mass: 1.0,
        charge: -1.0,
    };
    let particles = [electron];

    let energy_tol: F = 1e-3;
    psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t, energy_tol);

    // psi.normalization_by_1();
    // psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
}
