use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};

/// Трейт для SSFM
pub trait SSFM_ImaginaryTime {
    type WF;

    /// Эволюция на шаг по времени
    fn time_step_evol(&mut self, psi: &mut Self::WF, t: &mut Tspace);
}
