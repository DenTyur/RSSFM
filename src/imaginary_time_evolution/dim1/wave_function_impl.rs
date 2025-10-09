use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::common::units::AU_TO_EV;
use crate::config::{C, F};
use crate::dim1::{fft_maker::FftMaker1D, space::Xspace1D, wave_function::WaveFunction1D};
use crate::imaginary_time_evolution::dim1::ssfm_imaginary_time::SSFM1D_ImaginaryTime;
use crate::measure_time;
use crate::potentials::absorbing_potentials::absorbing_potential_1d;
use crate::potentials::potentials;
use crate::print_and_log;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use crate::traits::{fft_maker::FftMaker, wave_function::WaveFunction};
use rayon::prelude::*;
use std::env;
use std::fs;
use std::path::Path;

impl WaveFunction1D {
    pub fn evol_in_imag_time(
        &mut self,
        particles: &[Particle],
        atomic_pot: fn(x: [F; 1]) -> F,
        abs_pot: fn(x: [F; 1]) -> C,
        t: &mut Tspace,
    ) -> F {
        let mut ssfm_in_imaginary_time =
            SSFM1D_ImaginaryTime::new(&particles, &self.x, atomic_pot, abs_pot);

        // эволюция в мнимом времени
        let mut energy: F = compute_energy(self, atomic_pot).re * AU_TO_EV;

        print_and_log!("step\tenergy_last\tenergy_new\td_energy\ttime_per_step");
        let total_time = std::time::Instant::now();
        for i in 0..t.nt {
            let time_step = std::time::Instant::now();

            ssfm_in_imaginary_time.time_step_evol(self, t);
            self.normalization_by_1();

            let energy_current: F = compute_energy(self, atomic_pot).re * AU_TO_EV;
            let d_energy: F = (energy_current - energy).abs();

            print_and_log!(
                "{}/{}\t{} eV\t{} eV\t{} eV\t{} sec",
                i,
                t.nt,
                energy,
                energy_current,
                d_energy,
                time_step.elapsed().as_secs_f32()
            );

            energy = energy_current;

            if d_energy < 1e-5 {
                break;
            }
        }
        print_and_log!("Energy: {} eV", energy);
        print_and_log!("Total calculation time: {:?}", total_time.elapsed());
        energy
    }
}

// Вспомогательные функции для вычисления энергии
fn compute_energy(wf: &mut WaveFunction1D, atomic_pot: fn([F; 1]) -> F) -> C {
    compute_kinetic_energy(wf) + compute_potential_energy(wf, atomic_pot)
}

fn compute_potential_energy(wf: &mut WaveFunction1D, atomic_pot: fn([F; 1]) -> F) -> C {
    let mut potential_energy = C::new(0.0, 0.0);
    wf.psi
        .iter()
        .zip(wf.x.grid[0].iter())
        .for_each(|(psi, x)| potential_energy += psi.norm_sqr() * atomic_pot([*x]));
    potential_energy * wf.x.dx[0]
}

fn compute_kinetic_energy(wf: &mut WaveFunction1D) -> C {
    let mut fft_maker = FftMaker1D::new(&wf.x.n);
    fft_maker.modify_psi(wf);
    fft_maker.do_fft(wf);

    let mut kinetic_energy = C::new(0.0, 0.0);
    wf.psi.iter().zip(wf.p.grid[0].iter()).for_each(|(psi, p)| {
        kinetic_energy += psi.norm_sqr() * p * p / 2.0;
    });

    fft_maker.do_ifft(wf);
    fft_maker.demodify_psi(wf);
    kinetic_energy * wf.p.dp[0]
}
