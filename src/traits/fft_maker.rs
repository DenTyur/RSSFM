use crate::config::C;
use ndarray::{Array, Dimension};

/// Трейт для дискретного преобразования Фурье
pub trait FftMaker<D: Dimension> {
    type WaveFunctionDD;
    /// прямое преобразование фурье комплексного массива
    fn fft(&mut self, arr: &mut Array<C, D>);

    /// обратное преобразование фурье комплексного массива
    fn ifft(&mut self, arr: &mut Array<C, D>);

    /// модифицирует непрерывную psi для дискретного FFT
    fn modify_psi(&mut self, wf: &mut Self::WaveFunctionDD);

    /// демодифицирует дискретную "psi для DFT" обратно в непрерывную psi
    fn demodify_psi(&mut self, wf: &mut Self::WaveFunctionDD);
}
