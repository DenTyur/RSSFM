use super::space::{Pspace3D, Xspace3D};
use super::wave_function::WaveFunction3D;
use crate::config::{C, F, I, PI};
use crate::traits::fft_maker::FftMaker;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use rayon::prelude::*;

pub struct FftMaker3D {
    pub handler: [FftHandler<F>; 3],
    pub psi_temp: Array3<C>,
}

impl FftMaker3D {
    pub const DIM: usize = 3;

    pub fn new(n: &[usize; Self::DIM]) -> Self {
        let handler = [
            FftHandler::new(n[0]),
            FftHandler::new(n[1]),
            FftHandler::new(n[2]),
        ];
        let psi_temp: Array3<C> = Array::zeros((n[0], n[1], n[2]));
        Self { handler, psi_temp }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction3D) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        psi.psi = self.psi_temp.clone();
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction3D) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[2], 2);
        psi.psi = self.psi_temp.clone();
    }

    pub fn modify(&mut self, psi: &mut Array3<C>, x: &Xspace3D, p: &Pspace3D) {
        multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                *psi_elem *= x.dx[0] * x.dx[1] * x.dx[2] / (2. * PI).powf(3. / 2.)
                                    * (-I
                                        * (p.p0[0] * *x0_point
                                            + p.p0[1] * *x1_point
                                            + p.p0[2] * *x2_point))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }

    pub fn demodify(&mut self, psi: &mut Array3<C>, x: &Xspace3D, p: &Pspace3D) {
        multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), x.grid[2].iter())).for_each(
                            |(elem, x2_point)| {
                                *elem *= (2. * PI).powf(3. / 2.) / (x.dx[0] * x.dx[1] * x.dx[2])
                                    * (I * (p.p0[0] * *x0_point
                                        + p.p0[1] * *x1_point
                                        + p.p0[2] * *x2_point))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }
}

impl FftMaker<Ix3> for FftMaker3D {
    type WaveFunctionDD = WaveFunction3D;

    fn fft(&mut self, arr: &mut Array3<C>) {
        ndfft_par(&arr, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, arr, &mut self.handler[1], 1);
        ndfft_par(&arr, &mut self.psi_temp, &mut self.handler[2], 2);
        *arr = self.psi_temp.clone();
    }

    fn ifft(&mut self, arr: &mut Array3<C>) {
        ndifft_par(&arr, &mut self.psi_temp, &mut self.handler[0], 0);
        ndifft_par(&self.psi_temp, arr, &mut self.handler[1], 1);
        ndifft_par(&arr, &mut self.psi_temp, &mut self.handler[2], 2);
        *arr = self.psi_temp.clone();
    }

    fn modify_psi(&mut self, wf: &mut WaveFunction3D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                *psi_elem *= wf.x.dx[0] * wf.x.dx[1] * wf.x.dx[2]
                                    / (2. * PI).powf(3. / 2.)
                                    * (-I
                                        * (wf.p.p0[0] * *x0_point
                                            + wf.p.p0[1] * *x1_point
                                            + wf.p.p0[2] * *x2_point))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }

    fn demodify_psi(&mut self, wf: &mut WaveFunction3D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                *psi_elem *= (2. * PI).powf(3. / 2.)
                                    / (wf.x.dx[0] * wf.x.dx[1] * wf.x.dx[2])
                                    * (I * (wf.p.p0[0] * *x0_point
                                        + wf.p.p0[1] * *x1_point
                                        + wf.p.p0[2] * *x2_point))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }
}
