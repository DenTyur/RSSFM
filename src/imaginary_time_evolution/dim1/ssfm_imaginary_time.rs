use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::dim1::{fft_maker::FftMaker1D, space::Xspace1D, wave_function::WaveFunction1D};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use itertools::multizip;
use rayon::prelude::*;

#[allow(non_camel_case_types)]
pub struct SSFM1D_ImaginaryTime<'a, AP, AB> {
    particles: &'a [Particle],
    potential: AP,
    absorbing_potential: AB,
    fft_maker: FftMaker1D,
}

impl<'a, AP, AB> SSFM1D_ImaginaryTime<'a, AP, AB>
where
    AP: Fn([F; 1]) -> F + Send + Sync,
    AB: Fn([F; 1]) -> C + Send + Sync,
{
    pub fn new(
        particles: &'a [Particle],
        x: &Xspace1D,
        potential: AP,
        absorbing_potential: AB,
    ) -> Self {
        let fft_maker = FftMaker1D::new(&x.n);
        Self {
            particles,
            fft_maker,
            potential,
            absorbing_potential,
        }
    }
}

/// Реализация эволюции на временной шаг методом SSFM
impl<'a, AP, AB> SSFM_ImaginaryTime<1, AP, AB> for SSFM1D_ImaginaryTime<'a, AP, AB>
where
    AP: Fn([F; 1]) -> F + Send + Sync,
    AB: Fn([F; 1]) -> C + Send + Sync,
{
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (-0.5 * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn x_evol(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (-dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, dt: F) {
        let m = particles[0].mass;
        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-dt * (0.5 / m * px * px)).exp();
            });
    }

    fn time_step_evol(&mut self, wf: &mut WaveFunction1D, t: &mut Tspace) {
        self.fft_maker.modify_psi(wf);
        self.x_evol_half(
            self.particles,
            wf,
            t.dt,
            &self.potential,
            &self.absorbing_potential,
        );

        for _i in 0..t.n_steps - 1 {
            self.fft_maker.do_fft(wf);
            // Можно оптимизировать p_evol
            self.p_evol(self.particles, wf, t.dt);
            self.fft_maker.do_ifft(wf);
            self.x_evol(
                self.particles,
                wf,
                t.dt,
                &self.potential,
                &self.absorbing_potential,
            );
            t.current += t.dt;
        }

        self.fft_maker.do_fft(wf);
        self.p_evol(self.particles, wf, t.dt);

        self.fft_maker.do_ifft(wf);
        self.x_evol_half(
            self.particles,
            wf,
            t.dt,
            &self.potential,
            &self.absorbing_potential,
        );
        self.fft_maker.demodify_psi(wf);
        t.current += t.dt;
    }
}
