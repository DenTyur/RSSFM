use super::space::{Pspace4D, Xspace4D};
use super::wave_function::{Representation, WaveFunction4D};
use crate::config::{C, F, I, PI};
use crate::traits::fft_maker::FftMaker;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use rayon::prelude::*;

pub struct FftMaker4D {
    pub handler: [FftHandler<F>; 4],
    pub psi_temp: Array4<C>,
}

impl FftMaker4D {
    pub const DIM: usize = 4;

    pub fn new(n: &[usize; Self::DIM]) -> Self {
        let handler = [
            FftHandler::new(n[0]),
            FftHandler::new(n[1]),
            FftHandler::new(n[2]),
            FftHandler::new(n[3]),
        ];
        let psi_temp: Array4<C> = Array::zeros((n[0], n[1], n[2], n[3]));
        Self { handler, psi_temp }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction4D) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[3], 3);
        psi.representation = Representation::Momentum;
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction4D) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[3], 3);
        psi.representation = Representation::Position;
    }

    pub fn modify(&mut self, psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
        multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        *psi_elem *= x.dx[0] * x.dx[1] * x.dx[2] * x.dx[3]
                                            / (2. * PI).powi(2)
                                            * (-I
                                                * (p.p0[0] * *x0_point
                                                    + p.p0[1] * *x1_point
                                                    + p.p0[2] * *x2_point
                                                    + p.p0[3] * *x3_point))
                                                .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    pub fn demodify(&mut self, psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
        multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), x.grid[3].iter())).for_each(
                                    |(elem, x3_point)| {
                                        *elem *= (2. * PI).powi(2)
                                            / (x.dx[0] * x.dx[1] * x.dx[2] * x.dx[3])
                                            * (I * (p.p0[0] * *x0_point
                                                + p.p0[1] * *x1_point
                                                + p.p0[2] * *x2_point
                                                + p.p0[3] * *x3_point))
                                                .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }
}

impl FftMaker<Ix4> for FftMaker4D {
    type WaveFunctionDD = WaveFunction4D;

    fn fft(&mut self, arr: &mut Array4<C>) {
        ndfft_par(&arr, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, arr, &mut self.handler[1], 1);
        ndfft_par(&arr, &mut self.psi_temp, &mut self.handler[2], 2);
        ndfft_par(&self.psi_temp, arr, &mut self.handler[3], 3);
    }

    fn ifft(&mut self, arr: &mut Array4<C>) {
        ndifft_par(&arr, &mut self.psi_temp, &mut self.handler[0], 0);
        ndifft_par(&self.psi_temp, arr, &mut self.handler[1], 1);
        ndifft_par(&arr, &mut self.psi_temp, &mut self.handler[2], 2);
        ndifft_par(&self.psi_temp, arr, &mut self.handler[3], 3);
    }

    fn modify_psi(&mut self, wf: &mut WaveFunction4D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        *psi_elem *=
                                            wf.x.dx[0] * wf.x.dx[1] * wf.x.dx[2] * wf.x.dx[3]
                                                / (2. * PI).powi(2)
                                                * (-I
                                                    * (wf.p.p0[0] * *x0_point
                                                        + wf.p.p0[1] * *x1_point
                                                        + wf.p.p0[2] * *x2_point
                                                        + wf.p.p0[3] * *x3_point))
                                                    .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn demodify_psi(&mut self, wf: &mut WaveFunction4D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        *psi_elem *= (2. * PI).powi(2)
                                            / (wf.x.dx[0] * wf.x.dx[1] * wf.x.dx[2] * wf.x.dx[3])
                                            * (I * (wf.p.p0[0] * *x0_point
                                                + wf.p.p0[1] * *x1_point
                                                + wf.p.p0[2] * *x2_point
                                                + wf.p.p0[3] * *x3_point))
                                                .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }
}
