use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::common::units::AU_TO_EV;
use crate::config::{C, F};
use crate::dim2::{fft_maker::FftMaker2D, wave_function::WaveFunction2D};
use crate::imaginary_time_evolution::dim2::ssfm_imaginary_time::SSFM2D_ImaginaryTime;
use crate::print_and_log;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use crate::traits::{fft_maker::FftMaker, wave_function::WaveFunction};
use rayon::prelude::*;

/// Эволюция в мнимом времени для нахождения основного состояния и энергии
/// use rssfm::common::particle::Particle;
/// use rssfm::common::tspace::Tspace;
/// use rssfm::dim2::{space::Xspace2D, wave_function::WaveFunction2D};
/// use rssfm::potentials::{absorbing_potentials::absorbing_potential_2d, potentials};
/// use rssfm::F;
///
/// fn main() {
///     // let out_prefix = ".";
///     // let psi_path = "psi_initial.hdf5";
///
///     let mut t = Tspace::new(0., 0.01, 50, 500);
///
///     let x = Xspace2D::new([-30.0, -30.0], [0.5, 0.5], [120, 120]);
///     let mut psi = WaveFunction2D::init_oscillator_2d(x);
///
///     let abs_pot = |x: [F; 2]| absorbing_potential_2d(x, 50.0, 0.4);
///     let residual_charge: F = -1.0;
///     let smothed_parameter: F = 1.0;
///     let atomic_pot = |x: [F; 2]| potentials::soft_coulomb_2d(x, residual_charge, smothed_parameter);
///
///     let electron = Particle {
///         dim: 2,
///         mass: 1.0,
///         charge: -1.0,
///     };
///     let particles = [electron];
///
///     let energy_tol: F = 1e-5;
///     psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t, energy_tol);
///
///     // psi.normalization_by_1();
///     // psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
/// }
#[allow(clippy::needless_borrow)]
impl WaveFunction2D {
    pub fn evol_in_imag_time<AP, AB>(
        &mut self,
        particles: &[Particle],
        atomic_pot: AP,
        abs_pot: AB,
        t: &mut Tspace,
        energy_tol: F,
    ) -> F
    where
        AP: Fn([F; 2]) -> F + Send + Sync,
        AB: Fn([F; 2]) -> C + Send + Sync,
    {
        let mut ssfm_in_imaginary_time =
            SSFM2D_ImaginaryTime::new(&particles, &self.x, &atomic_pot, abs_pot);

        // эволюция в мнимом времени
        let mut energy: F = compute_energy(self, &atomic_pot, &particles) * AU_TO_EV;

        print_and_log!(
            "{:>8} {:>18} {:>18} {:>18} {:>12}",
            "step",
            "  energy_last eV",
            "  energy_new eV",
            "    d_energy eV",
            "  time_sec"
        );

        let total_time = std::time::Instant::now();
        for i in 0..t.nt {
            let time_step = std::time::Instant::now();

            ssfm_in_imaginary_time.time_step_evol(self, t);
            self.normalization_by_1();

            let energy_current: F = compute_energy(self, &atomic_pot, &particles) * AU_TO_EV;
            let d_energy: F = (energy_current - energy).abs();

            print_and_log!(
                "{:>8} {:>18.10} {:>18.10} {:>18.10} {:>12.6}",
                format!("{}/{}", i, t.nt),
                energy,
                energy_current,
                d_energy,
                time_step.elapsed().as_secs_f32()
            );

            energy = energy_current;

            if d_energy < energy_tol {
                break;
            }
        }
        print_and_log!("Energy: {} eV", energy);
        print_and_log!("Total calculation time: {:?}", total_time.elapsed());
        energy
    }
}

// Вспомогательные функции для вычисления энергии
fn compute_energy<AP>(wf: &mut WaveFunction2D, atomic_pot: &AP, particles: &[Particle]) -> F
where
    AP: Fn([F; 2]) -> F + Send + Sync,
{
    compute_kinetic_energy(wf, particles) + compute_potential_energy(wf, atomic_pot)
}

fn compute_potential_energy<AP>(wf: &mut WaveFunction2D, atomic_pot: &AP) -> F
where
    AP: Fn([F; 2]) -> F + Send + Sync,
{
    let potential_energy: F = wf
        .psi
        .indexed_iter()
        .par_bridge()
        .map(|((i, j), psi)| {
            let x = wf.x.grid[0][i];
            let y = wf.x.grid[1][j];
            psi.norm_sqr() * atomic_pot([x, y])
        })
        .sum::<F>();

    potential_energy * wf.x.dx[0] * wf.x.dx[1]
}

fn compute_kinetic_energy(wf: &mut WaveFunction2D, particles: &[Particle]) -> F {
    // переходим в импульсное представление
    let mut fft_maker = FftMaker2D::new(&wf.x.n);
    fft_maker.modify_psi(wf);
    fft_maker.do_fft(wf);

    let [m0, m1] = match particles.len() {
        1 => [particles[0].mass, particles[0].mass],
        2 => [particles[0].mass, particles[1].mass],
        _ => panic!("Неправильная размерность particles"),
    };

    let kinetic_energy: F = wf
        .psi
        .indexed_iter()
        .par_bridge()
        .map(|((i, j), psi)| {
            let p0 = wf.p.grid[0][i];
            let p1 = wf.p.grid[1][j];
            psi.norm_sqr() * (p0 * p0 / (2.0 * m0) + p1 * p1 / (2.0 * m1))
        })
        .sum::<F>();

    // переходим обратно в координатное представление
    fft_maker.do_ifft(wf);
    fft_maker.demodify_psi(wf);
    kinetic_energy * wf.p.dp[0] * wf.p.dp[1]
}
