use crate::field;
use crate::gauge;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field2D;
use gauge::{LenthGauge, VelocityGauge};
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::{br_1e2d_external, AtomicPotential};
use rayon::prelude::*;
use std::f32::consts::PI;
use wave_function::WaveFunction;

type F = f32;
type C = Complex<f32>;
const I: C = Complex::I;

pub trait EvolutionSSFM {
    fn x_evol_half(
        &self,
        psi: &mut WaveFunction,
        x: &Xspace,
        t: &Tspace,
        potential: fn(x: F, y: F) -> F,
        absorbing_potential: fn(x: F, y: F) -> C,
    );
    fn x_evol(
        &self,
        psi: &mut WaveFunction,
        x: &Xspace,
        t: &Tspace,
        potential: fn(x: F, y: F) -> F,
        absorbing_potential: fn(x: F, y: F) -> C,
    );
    fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace);
}

impl<'a> EvolutionSSFM for VelocityGauge<'a> {
    fn x_evol_half(
        &self,
        psi: &mut WaveFunction,
        x: &Xspace,
        t: &Tspace,
        potential: fn(x: F, y: F) -> F,
        absorbing_potential: fn(x: F, y: F) -> C,
    ) {
        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter())).for_each(|(psi_elem, y_point)| {
                    let potential_elem = potential(*x_point, *y_point);
                    let absorbing_potential_elem = absorbing_potential(*x_point, *y_point);
                    *psi_elem *=
                        (-I * 0.5 * t.dt * (potential_elem + absorbing_potential_elem)).exp();
                });
            });
    }

    fn x_evol(
        &self,
        psi: &mut WaveFunction,
        x: &Xspace,
        t: &Tspace,
        potential: fn(x: F, y: F) -> F,
        absorbing_potential: fn(x: F, y: F) -> C,
    ) {
        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter())).for_each(|(psi_elem, y_point)| {
                    let potential_elem = potential(*x_point, *y_point);
                    let absorbing_potential_elem = absorbing_potential(*x_point, *y_point);
                    *psi_elem *= (-I * t.dt * (potential_elem + absorbing_potential_elem)).exp();
                });
            });
    }

    fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace) {
        let vec_pot = self.field.vec_pot(t.current);

        multizip((psi.psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I
                            * t.dt
                            * (0.5 * px * px + 0.5 * py * py + vec_pot[0] * px + vec_pot[1] * py))
                            .exp();
                    });
            });
    }
}

// impl<'a> EvolutionSSFM for LenthGauge<'a> {
//     fn x_evol_half(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
//         let electric_field = self.field.electric_field_time_dependence(t.current);
//
//         multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
//             .par_bridge()
//             .for_each(|(mut psi_row, x_point)| {
//                 multizip((psi_row.iter_mut(), x.grid[1].iter())).for_each(|(psi_elem, y_point)| {
//                     let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
//                     *psi_elem *= (-I
//                         * 0.5
//                         * t.dt
//                         * (atomic_potential_elem
//                                 + electric_field[0] * x_point // заряд у электрона минус, поэтому
//                             // здесь плюс
//                                 + electric_field[1] * y_point))
//                         .exp();
//                 });
//             });
//     }
//
//     fn x_evol(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
//         let electric_field = self.field.electric_field_time_dependence(t.current);
//
//         multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
//             .par_bridge()
//             .for_each(|(mut psi_row, x_point)| {
//                 multizip((psi_row.iter_mut(), x.grid[1].iter())).for_each(|(psi_elem, y_point)| {
//                     let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
//                     *psi_elem *= (-I
//                         * t.dt
//                         * (atomic_potential_elem
//                                 + electric_field[0] * x_point // заряд у электрона минус, поэтому
//                             // здесь плюс
//                                 + electric_field[1] * y_point))
//                         .exp();
//                 });
//             });
//     }
//
//     fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace) {
//         multizip((psi.psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
//             .par_bridge()
//             .for_each(|(mut psi_row, px)| {
//                 psi_row
//                     .iter_mut()
//                     .zip(p.grid[1].iter())
//                     .for_each(|(psi_elem, py)| {
//                         *psi_elem *= (-I * t.dt * (0.5 * px * px + 0.5 * py * py)).exp();
//                     });
//             });
//     }
// }

pub struct SSFM<'a, G: EvolutionSSFM> {
    potential: fn(x: F, y: F) -> F,
    absorbing_potential: fn(x: F, y: F) -> C,
    gauge: &'a G,
    fft: FftMaker2d,
    x: &'a Xspace,
    p: &'a Pspace,
}

impl<'a, G: EvolutionSSFM> SSFM<'a, G> {
    pub fn new(
        gauge: &'a G,
        x: &'a Xspace,
        p: &'a Pspace,
        potential: fn(x: F, y: F) -> F,
        absorbing_potential: fn(x: F, y: F) -> C,
    ) -> Self {
        let fft = FftMaker2d::new(&x.n);
        Self {
            gauge,
            fft,
            x,
            p,
            potential,
            absorbing_potential,
        }
    }

    pub fn demodify_psi(&mut self, psi: &mut WaveFunction) {
        // демодифицирует "psi для DFT" обратно в psi
        multizip((psi.psi.axis_iter_mut(Axis(0)), self.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), self.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        *psi_elem *= (2. * PI) / (self.x.dx[0] * self.x.dx[1])
                            * (I * (self.p.p0[0] * x_point + self.p.p0[1] * y_point)).exp();
                    },
                );
            });
    }

    pub fn modify_psi(&mut self, psi: &mut WaveFunction) {
        // модифицирует psi для FFT
        multizip((psi.psi.axis_iter_mut(Axis(0)), self.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), self.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        *psi_elem *= self.x.dx[0] * self.x.dx[1] / (2. * PI)
                            * (-I * (self.p.p0[0] * x_point + self.p.p0[1] * *y_point)).exp();
                    },
                );
            });
    }

    pub fn time_step_evol(
        &mut self,
        psi: &mut WaveFunction,
        t: &mut Tspace,
        psi_x_save_path: Option<&str>,
        psi_p_save_path: Option<&str>,
    ) {
        if let Some(path) = psi_x_save_path {
            psi.save_psi(path).unwrap();
        }
        self.modify_psi(psi);
        self.gauge
            .x_evol_half(psi, self.x, t, self.potential, self.absorbing_potential);

        for _i in 0..t.n_steps - 1 {
            self.fft.do_fft(psi);
            // Можно оптимизировать p_evol
            self.gauge.p_evol(psi, self.p, t);
            self.fft.do_ifft(psi);
            self.gauge
                .x_evol(psi, self.x, t, self.potential, self.absorbing_potential);
            t.current += t.dt;
        }

        self.fft.do_fft(psi);
        self.gauge.p_evol(psi, self.p, t);
        if let Some(path) = psi_p_save_path {
            psi.save_psi(path).unwrap();
        }
        self.fft.do_ifft(psi);
        self.gauge
            .x_evol_half(psi, self.x, t, self.potential, self.absorbing_potential);
        self.demodify_psi(psi);
        t.current += t.dt;
    }
}

pub struct FftMaker2d {
    pub handler: Vec<FftHandler<F>>,
    pub psi_temp: Array2<C>,
}

impl FftMaker2d {
    pub fn new(n: &Vec<usize>) -> Self {
        Self {
            handler: Vec::from_iter(0..n.len()) // тоже костыль! как это сделать через функцию?
                .iter()
                .map(|&i| FftHandler::new(n[i]))
                .collect(),
            psi_temp: Array::zeros((n[0], n[1])),
        }
    }

    pub fn fft(&mut self, psi: &mut Array2<C>) {
        ndfft_par(psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, psi, &mut self.handler[1], 1);
    }
    pub fn ifft(&mut self, psi: &mut Array2<C>) {
        ndifft_par(psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, psi, &mut self.handler[0], 0);
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
    }
}
