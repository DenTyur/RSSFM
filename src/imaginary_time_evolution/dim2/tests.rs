use rssfm::common::particle::Particle;
use rssfm::common::tspace::Tspace;
use rssfm::dim2::{space::Xspace2D, wave_function::WaveFunction2D};
use rssfm::potentials::{absorbing_potentials::absorbing_potential_2d, potentials};
use rssfm::F;

fn main() {
    // let out_prefix = ".";
    // let psi_path = "psi_initial.hdf5";

    let mut t = Tspace::new(0., 0.01, 50, 500);

    let x = Xspace2D::new([-30.0, -30.0], [0.5, 0.5], [120, 120]);
    let mut psi = WaveFunction2D::init_oscillator_2d(x);

    let abs_pot = |x: [F; 2]| absorbing_potential_2d(x, 50.0, 0.4);
    let residual_charge: F = -1.0;
    let smothed_parameter: F = 1.0;
    let atomic_pot = |x: [F; 2]| potentials::soft_coulomb_2d(x, residual_charge, smothed_parameter);

    let electron = Particle {
        dim: 2,
        mass: 1.0,
        charge: -1.0,
    };
    let particles = [electron];

    let energy_tol: F = 1e-5;
    psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t, energy_tol);

    // psi.normalization_by_1();
    // psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
}
