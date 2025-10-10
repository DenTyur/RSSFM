use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::wave_function::WaveFunction;

/// Трейт для SSFM
#[allow(non_camel_case_types)]
pub trait SSFM_ImaginaryTime<const D: usize, AP, AB>
where
    AP: Fn([F; D]) -> F + Send + Sync, // функция для атомного потенциала
    AB: Fn([F; D]) -> C + Send + Sync,
{
    type WF: WaveFunction<D>;

    /// Эволюция на шаг по времени
    fn time_step_evol(&mut self, psi: &mut Self::WF, t: &mut Tspace);

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut Self::WF,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    );

    fn x_evol(
        &self,
        particles: &[Particle],
        wf: &mut Self::WF,
        dt: F,
        potential: &AP,
        absorbing_potential: &AB,
    );

    fn p_evol(&self, particles: &[Particle], wf: &mut Self::WF, dt: F);
}
