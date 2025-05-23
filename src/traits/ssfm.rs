use crate::common::tspace::Tspace;
use crate::config::{C, F};

/// Трейт для операторов эволюции для SSFM в разных калибровках
pub trait GaugedEvolutionSSFM<const D: usize> {
    type WF;

    /// Действите оператором эволюции в координатном пространстве
    /// на половину временного шага
    fn x_evol_half(
        &self,
        psi: &mut Self::WF,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; D]) -> F,
        absorbing_potential: fn(x: [F; D]) -> C,
    );

    /// Действите оператором эволюции в координатном пространстве
    /// на полный временной шаг
    fn x_evol(
        &self,
        psi: &mut Self::WF,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; D]) -> F,
        absorbing_potential: fn(x: [F; D]) -> C,
    );

    /// Действите оператором эволюции в импульсном пространстве
    fn p_evol(&self, psi: &mut Self::WF, tcurrent: F, dt: F);
}

/// Трейт для SSFM
pub trait SSFM {
    type WF;

    /// Эволюция на шаг по времени
    fn time_step_evol(&mut self, psi: &mut Self::WF, t: &mut Tspace, psi_p_save_path: Option<&str>);
}
