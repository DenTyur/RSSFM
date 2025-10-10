use rssfm::common::particle::Particle;
use rssfm::common::tspace::Tspace;
use rssfm::dim1::{space::Xspace1D, wave_function::WaveFunction1D};
use rssfm::potentials::{absorbing_potentials::absorbing_potential_1d, potentials};
use rssfm::F;

fn main() {
    // let out_prefix = ".";
    // let psi_path = "psi_initial.hdf5";

    let mut t = Tspace::new(0., 0.01, 50, 500);

    let x = Xspace1D::new([-30.0], [0.5], [120]);
    let mut psi = WaveFunction1D::init_oscillator_1d(x);

    let abs_pot = |x: [F; 1]| absorbing_potential_1d(x, 50.0, 0.4);
    let atomic_pot = |x: [F; 1]| potentials::soft_coulomb_1d(x, -1.0, 1.66);

    let electron1 = Particle {
        dim: 1,
        mass: 1.0,
        charge: -1.0,
    };
    let particles = [electron1];

    let energy_tol: F = 1e-5;
    psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t, energy_tol);

    // psi.normalization_by_1();
    // psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
}
