use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::dim4::{fft_maker::FftMaker4D, space::Xspace4D, wave_function::WaveFunction4D};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm_imaginary_time::SSFM_ImaginaryTime;
use itertools::multizip;
use ndarray::prelude::*;
use rayon::prelude::*;

#[allow(non_camel_case_types)]
pub struct SSFM4D_ImaginaryTime<'a, AP, AB> {
    particles: &'a [Particle],
    potential: AP,
    absorbing_potential: AB,
    fft_maker: FftMaker4D,
}

impl<'a, AP, AB> SSFM4D_ImaginaryTime<'a, AP, AB>
where
    AP: Fn([F; 4]) -> F + Send + Sync,
    AB: Fn([F; 4]) -> C + Send + Sync,
{
    pub fn new(
        particles: &'a [Particle],
        x: &Xspace4D,
        potential: AP,
        absorbing_potential: AB,
    ) -> Self {
        let fft_maker = FftMaker4D::new(&x.n);
        Self {
            particles,
            fft_maker,
            potential,
            absorbing_potential,
        }
    }
}

/// Реализация эволюции на временной шаг методом SSFM
impl<'a, AP, AB> SSFM_ImaginaryTime<4, AP, AB> for SSFM4D_ImaginaryTime<'a, AP, AB>
where
    AP: Fn([F; 4]) -> F + Send + Sync,
    AB: Fn([F; 4]) -> C + Send + Sync,
{
    type WF = WaveFunction4D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction4D,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        let potential_elem =
                                            potential([*x0_point, *x1_point, *x2_point, *x3_point]);
                                        let absorbing_potential_elem = absorbing_potential([
                                            *x0_point, *x1_point, *x2_point, *x3_point,
                                        ]);
                                        *psi_elem *= (-0.5
                                            * dt
                                            * (potential_elem + absorbing_potential_elem))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn x_evol(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction4D,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        let potential_elem =
                                            potential([*x0_point, *x1_point, *x2_point, *x3_point]);
                                        let absorbing_potential_elem = absorbing_potential([
                                            *x0_point, *x1_point, *x2_point, *x3_point,
                                        ]);
                                        *psi_elem *= (-dt
                                            * (potential_elem + absorbing_potential_elem))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction4D, dt: F) {
        let [m0, m1] = match particles.len() {
            1 => [particles[0].mass, particles[0].mass],
            2 => [particles[0].mass, particles[1].mass],
            _ => panic!("Неправильная размерность particles"),
        };

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, p0)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.p.grid[1].iter())).for_each(
                    |(mut psi_2d, p1)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.p.grid[2].iter())).for_each(
                            |(mut psi_1d, p2)| {
                                multizip((psi_1d.iter_mut(), wf.p.grid[3].iter())).for_each(
                                    |(psi_elem, p3)| {
                                        *psi_elem *= (-dt
                                            * (0.5 / m0 * p0 * p0
                                                + 0.5 / m0 * p1 * p1
                                                + 0.5 / m1 * p2 * p2
                                                + 0.5 / m1 * p3 * p3))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn time_step_evol(&mut self, wf: &mut WaveFunction4D, t: &mut Tspace) {
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
