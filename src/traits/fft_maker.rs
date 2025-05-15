use crate::config::C;
use ndarray::{Array, Dimension};

pub trait FftMaker<D: Dimension> {
    type WaveFunctionDD;

    /// прямое преобразование фурье комплексного массива
    fn fft(&mut self, arr: &mut Array<C, D>);
    /// обратное преобразование фурье комплексного массива
    fn ifft(&mut self, arr: &mut Array<C, D>);
    /// модифицирует psi для FFT
    fn modify_psi(&mut self, wf: &mut Self::WaveFunctionDD);
    /// демодифицирует "psi для DFT" обратно в psi
    fn demodify_psi(&mut self, wf: &mut Self::WaveFunctionDD);
}
