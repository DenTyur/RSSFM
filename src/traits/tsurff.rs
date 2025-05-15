use crate::common::tspace::Tspace;
use crate::config::F;
use crate::traits::{
    flow::{Flux, SurfaceFlow},
    volkov::VolkovGauge,
    wave_function::ValueAndSpaceDerivatives,
};
use std::marker::{Send, Sync};

/// Трейт для вычисления импульсного распределения методом t-surff.
/// Есть зависимость от:
/// 1) Калибровки (gauge)
/// 2) Поверхности, через которую считается поток (surface)
/// 3) Также передается координатная и импульсная сетки x и p_true
/// 4) Если импульсная сетка больше, чем присутствующие импульсы, ее можно обрезать параметром p_cut
pub trait Tsurff<'a, const D: usize, G, S>
where
    G: VolkovGauge + Flux<D> + Send + Sync + Copy,
    S: SurfaceFlow<D, G> + Sync,
{
    type Pspace;
    fn new(gauge: &'a G, surface: &'a S, p_true: &Self::Pspace, pcut: Option<F>) -> Self;
    fn save_as_npy(&self, path: &str);
    fn plot(&self, path: &str, colorbar_limits: [F; 2]);
    fn plot_log(&self, path: &str, colorbar_limits: [F; 2]);
    fn time_integration_step(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives<D> + Send + Sync),
        t: &Tspace,
    );
}
