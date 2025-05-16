use super::wave_function::WaveFunction2D;
use crate::config::{C, F, I, PI};
use crate::traits::fft_maker::FftMaker;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use rayon::prelude::*;

pub struct FftMaker2D {
    pub handler: [FftHandler<F>; 2],
    pub psi_temp: Array2<C>,
}

impl FftMaker2D {
    pub const DIM: usize = 2;

    pub fn new(n: &[usize; Self::DIM]) -> Self {
        let handler = [FftHandler::new(n[0]), FftHandler::new(n[1])];
        let psi_temp: Array2<C> = Array::zeros((n[0], n[1]));
        Self { handler, psi_temp }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction2D) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction2D) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
        // ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        // ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
    }
}

impl FftMaker<Ix2> for FftMaker2D {
    type WaveFunctionDD = WaveFunction2D;

    fn fft(&mut self, arr: &mut Array2<C>) {
        ndfft_par(arr, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, arr, &mut self.handler[1], 1);
    }

    fn ifft(&mut self, arr: &mut Array2<C>) {
        ndifft_par(arr, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, arr, &mut self.handler[0], 0);
    }

    fn modify_psi(&mut self, wf: &mut WaveFunction2D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), wf.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        *psi_elem *= wf.x.dx[0] * wf.x.dx[1] / (2. * PI)
                            * (-I * (wf.p.p0[0] * x_point + wf.p.p0[1] * *y_point)).exp();
                    },
                );
            });
    }

    fn demodify_psi(&mut self, wf: &mut WaveFunction2D) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), wf.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        *psi_elem *= (2. * PI) / (wf.x.dx[0] * wf.x.dx[1])
                            * (I * (wf.p.p0[0] * x_point + wf.p.p0[1] * y_point)).exp();
                    },
                );
            });
    }
}
