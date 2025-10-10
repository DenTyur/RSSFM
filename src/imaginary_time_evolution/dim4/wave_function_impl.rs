use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::common::units::AU_TO_EV;
use crate::config::{C, F};
use crate::dim4::{fft_maker::FftMaker4D, wave_function::WaveFunction4D};
use crate::imaginary_time_evolution::dim4::ssfm_imaginary_time::SSFM4D_ImaginaryTime;
use crate::print_and_log;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use crate::traits::{fft_maker::FftMaker, wave_function::WaveFunction};
use rayon::prelude::*;

/// Эволюция в мнимом времени для нахождения основного состояния и энергии
/// EXAMPLE:
/// use rssfm::common::particle::Particle;
/// use rssfm::common::tspace::Tspace;
/// use rssfm::dim4::{space::Xspace4D, wave_function::WaveFunction4D};
/// use rssfm::potentials::{absorbing_potentials::absorbing_potential_4d, potentials};
/// use rssfm::traits::wave_function::WaveFunction;
/// use rssfm::F;
///
/// fn main() {
///     // let out_prefix = ".";
///     // let psi_path = "psi_initial.hdf5";
///
///     let mut t = Tspace::new(0., 0.01, 1, 10000);
///
///     let x = Xspace4D::new(
///         [-15.0, -15.0, -15.0, -15.0],
///         [0.5, 0.5, 0.5, 0.5],
///         [64, 64, 64, 64],
///     );
///     // let mut psi = WaveFunction4D::init_oscillator_4d(x);
///     let mut psi = WaveFunction4D::init_from_hdf5(
///         "/home/denis/Programs/atoms_and_ions/DATA/br/br2e2d_N64_dx05_interact.hdf5",
///     );
///     psi.extend(&x);
///
///     let abs_pot = |x: [F; 4]| absorbing_potential_4d(x, 50.0, 0.4);
///     let atomic_pot = |x: [F; 4]| potentials::br_2e2d(x);
///
///     let electron = Particle {
///         dim: 2,
///         mass: 1.0,
///         charge: -1.0,
///     };
///     let particles = [electron];
///
///     let energy_tol: F = 1e-3;
///     psi.evol_in_imag_time(&particles, atomic_pot, abs_pot, &mut t, energy_tol);
///
///     // psi.normalization_by_1();
///     // psi.save_as_hdf5(format!("{}/{}", out_prefix, psi_path).as_str());
/// }
#[allow(clippy::needless_borrow)]
impl WaveFunction4D {
    pub fn evol_in_imag_time<AP, AB>(
        &mut self,
        particles: &[Particle],
        atomic_pot: AP,
        abs_pot: AB,
        t: &mut Tspace,
        energy_tol: F,
    ) -> F
    where
        AP: Fn([F; 4]) -> F + Send + Sync,
        AB: Fn([F; 4]) -> C + Send + Sync,
    {
        let mut ssfm_in_imaginary_time =
            SSFM4D_ImaginaryTime::new(&particles, &self.x, &atomic_pot, abs_pot);

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
fn compute_energy<AP>(wf: &mut WaveFunction4D, atomic_pot: &AP, particles: &[Particle]) -> F
where
    AP: Fn([F; 4]) -> F + Send + Sync,
{
    compute_kinetic_energy(wf, particles) + compute_potential_energy(wf, atomic_pot)
}

fn compute_potential_energy<AP>(wf: &mut WaveFunction4D, atomic_pot: &AP) -> F
where
    AP: Fn([F; 4]) -> F + Send + Sync,
{
    let potential_energy: F = wf
        .psi
        .indexed_iter()
        .par_bridge()
        .map(|((i0, i1, i2, i3), psi)| {
            let x0 = wf.x.grid[0][i0];
            let x1 = wf.x.grid[1][i1];
            let x2 = wf.x.grid[2][i2];
            let x3 = wf.x.grid[3][i3];
            psi.norm_sqr() * atomic_pot([x0, x1, x2, x3])
        })
        .sum::<F>();

    potential_energy * wf.x.dx[0] * wf.x.dx[1] * wf.x.dx[2] * wf.x.dx[3]
}

fn compute_kinetic_energy(wf: &mut WaveFunction4D, particles: &[Particle]) -> F {
    // переходим в импульсное представление
    let mut fft_maker = FftMaker4D::new(&wf.x.n);
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
        .map(|((i0, i1, i2, i3), psi)| {
            let p0 = wf.p.grid[0][i0];
            let p1 = wf.p.grid[1][i1];
            let p2 = wf.p.grid[2][i2];
            let p3 = wf.p.grid[3][i3];
            psi.norm_sqr() * ((p0 * p0 + p1 * p1) / (2.0 * m0) + (p2 * p2 + p3 * p3) / (2.0 * m1))
        })
        .sum::<F>();

    // переходим обратно в координатное представление
    fft_maker.do_ifft(wf);
    fft_maker.demodify_psi(wf);
    kinetic_energy * wf.p.dp[0] * wf.p.dp[1] * wf.p.dp[2] * wf.p.dp[3]
}
