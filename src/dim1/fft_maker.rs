use super::wave_function::WaveFunction1D;
use crate::config::{C, F, I, PI};
use crate::traits::fft_maker::FftMaker;
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use rayon::prelude::*;

pub struct FftMaker1D {
    pub handler: [FftHandler<F>; 1],
    pub psi_temp: Array1<C>,
}

impl FftMaker1D {
    pub const DIM: usize = 1;

    pub fn new(n: &[usize; Self::DIM]) -> Self {
        let handler = [FftHandler::new(n[0])];
        let psi_temp: Array1<C> = Array::zeros(n[0]);
        Self { handler, psi_temp }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction1D) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        psi.psi = self.psi_temp.clone();
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction1D) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        psi.psi = self.psi_temp.clone();
    }
}

impl FftMaker<Ix1> for FftMaker1D {
    type WaveFunctionDD = WaveFunction1D;

    fn fft(&mut self, arr: &mut Array1<C>) {
        ndfft_par(arr, &mut self.psi_temp, &mut self.handler[0], 0);
        *arr = self.psi_temp.clone();
    }

    fn ifft(&mut self, arr: &mut Array1<C>) {
        ndifft_par(arr, &mut self.psi_temp, &mut self.handler[0], 0);
        *arr = self.psi_temp.clone();
    }

    fn modify_psi(&mut self, wf: &mut WaveFunction1D) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_elem, x_point)| {
                *psi_elem *= wf.x.dx[0] / (2. * PI).sqrt() * (-I * (wf.p.p0[0] * *x_point)).exp();
            });
    }

    fn demodify_psi(&mut self, wf: &mut WaveFunction1D) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_elem, x_point)| {
                *psi_elem *= (2. * PI).sqrt() / wf.x.dx[0] * (I * (wf.p.p0[0] * x_point)).exp();
            });
    }
}
