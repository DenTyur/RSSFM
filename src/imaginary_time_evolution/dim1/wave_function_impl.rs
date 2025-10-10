use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::common::units::AU_TO_EV;
use crate::config::{C, F};
use crate::dim1::{fft_maker::FftMaker1D, wave_function::WaveFunction1D};
use crate::imaginary_time_evolution::dim1::ssfm_imaginary_time::SSFM1D_ImaginaryTime;
use crate::print_and_log;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use crate::traits::{fft_maker::FftMaker, wave_function::WaveFunction};

/// Эволюция в мнимом времени для нахождения основного состояния и энергии
/// EXAMPLE:
/// use rssfm::common::particle::Particle;
/// use rssfm::common::tspace::Tspace;
/// use rssfm::dim1::{space::Xspace1D, wave_function::WaveFunction1D};
/// use rssfm::potentials::{absorbing_potentials::absorbing_potential_1d, potentials};
/// use rssfm::traits::wave_function::WaveFunction;
/// use rssfm::F;
///
/// fn main() {
///     let out_prefix = ".";
///     let psi_path = "psi_initial.hdf5";
///
///     let mut t = Tspace::new(0., 0.01, 50, 500);
///
///     let x = Xspace1D::new([-30.0], [0.5], [120]);
///     let mut psi = WaveFunction1D::init_oscillator_1d(x);
///
///     let abs_pot = |x: [F; 1]| absorbing_potential_1d(x, 50.0, 0.4);
///     let atomic_pot = |x: [F; 1]| potentials::soft_coulomb_1d(x, -1.0, 1.66);
///
///     let electron1 = Particle {
///         dim: 1,
///         mass: 1.0,
///         charge: -1.0,
///     };
///     let particles = [electron1];
///
///     psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t);
///
///     psi.normalization_by_1();
///     psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
/// }
#[allow(clippy::needless_borrow)]
impl WaveFunction1D {
    pub fn evol_in_imag_time<AP, AB>(
        &mut self,
        particles: &[Particle],
        atomic_pot: AP,
        abs_pot: AB,
        t: &mut Tspace,
        energy_tol: F,
    ) -> F
    where
        AP: Fn([F; 1]) -> F + Send + Sync,
        AB: Fn([F; 1]) -> C + Send + Sync,
    {
        let mut ssfm_in_imaginary_time =
            SSFM1D_ImaginaryTime::new(&particles, &self.x, &atomic_pot, abs_pot);

        // эволюция в мнимом времени
        let mut energy: F = compute_energy(self, &atomic_pot, particles) * AU_TO_EV;

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

            let energy_current: F = compute_energy(self, &atomic_pot, particles) * AU_TO_EV;
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
fn compute_energy<AP>(wf: &mut WaveFunction1D, atomic_pot: &AP, particles: &[Particle]) -> F
where
    AP: Fn([F; 1]) -> F,
{
    compute_kinetic_energy(wf, particles) + compute_potential_energy(wf, atomic_pot)
}

fn compute_potential_energy<AP>(wf: &mut WaveFunction1D, atomic_pot: &AP) -> F
where
    AP: Fn([F; 1]) -> F,
{
    let mut potential_energy: F = 0.0;
    wf.psi
        .iter()
        .zip(wf.x.grid[0].iter())
        .for_each(|(psi, x)| potential_energy += psi.norm_sqr() * atomic_pot([*x]));
    potential_energy * wf.x.dx[0]
}

fn compute_kinetic_energy(wf: &mut WaveFunction1D, particles: &[Particle]) -> F {
    let mut fft_maker = FftMaker1D::new(&wf.x.n);
    fft_maker.modify_psi(wf);
    fft_maker.do_fft(wf);

    let mut kinetic_energy: F = 0.0;

    let m = match particles.len() {
        1 => particles[0].mass,
        _ => panic!("Неправильная размерность particles"),
    };

    wf.psi.iter().zip(wf.p.grid[0].iter()).for_each(|(psi, p)| {
        kinetic_energy += psi.norm_sqr() * p * p / (2.0 * m);
    });

    fft_maker.do_ifft(wf);
    fft_maker.demodify_psi(wf);
    kinetic_energy * wf.p.dp[0]
}
