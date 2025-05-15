use crate::config::{C, F};
use crate::traits::wave_function::ValueAndSpaceDerivatives;
use std::marker::{Send, Sync};

/// Трейт для вычисления потока вероятности через поверхность
/// Применим к поверхности, через которую вычисляется поток
pub trait SurfaceFlow<const D: usize, G: Flux<D> + Send + Sync> {
    /// Возвращает поток вероятности через поверхность
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives<D> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<D> + Send + Sync),
        t: F,
    ) -> C;
}

/// Вычисляет вектор плотности потока вероятности
pub trait Flux<const D: usize> {
    /// Возвращает вектор j в точке x
    fn compute_flux(
        &self,
        x: [F; D],
        psi1: &(impl ValueAndSpaceDerivatives<D> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<D> + Send + Sync),
        t: F,
    ) -> [C; D];
}
