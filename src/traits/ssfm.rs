use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};

/// Трейт для операторов эволюции для SSFM в разных калибровках для одной частицы
pub trait GaugedEvolutionSSFM<const D: usize> {
    type WF;

    /// Действите оператором эволюции в координатном пространстве
    /// на половину временного шага
    fn x_evol_half(
        &self,
        particles: &[Particle],
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
        particles: &[Particle],
        psi: &mut Self::WF,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; D]) -> F,
        absorbing_potential: fn(x: [F; D]) -> C,
    );

    /// Действите оператором эволюции в импульсном пространстве
    fn p_evol(&self, particles: &[Particle], psi: &mut Self::WF, tcurrent: F, dt: F);
}

/// Трейт для операторов эволюции для SSFM в разных калибровках для двух частиц
pub trait GaugedEvolutionSSFMtwoParticles<const D: usize> {
    // D -- полная размерность
    type WF;

    /// Действите оператором эволюции в координатном пространстве
    /// на половину временного шага
    fn x_evol_half_2particles(
        &self,
        particles: &[Particle; 2],
        psi: &mut Self::WF,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; D]) -> F,
        absorbing_potential: fn(x: [F; D]) -> C,
    );

    /// Действите оператором эволюции в координатном пространстве
    /// на полный временной шаг
    fn x_evol_2particles(
        &self,
        particles: &[Particle; 2],
        psi: &mut Self::WF,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; D]) -> F,
        absorbing_potential: fn(x: [F; D]) -> C,
    );

    /// Действите оператором эволюции в импульсном пространстве
    fn p_evol_2particles(&self, particles: &[Particle; 2], psi: &mut Self::WF, tcurrent: F, dt: F);
}

/// Трейт для SSFM
pub trait SSFM {
    type WF;

    /// Эволюция на шаг по времени
    fn time_step_evol(
        &mut self,
        psi: &mut Self::WF,
        t: &mut Tspace,
        psi_p_save_path: Option<(&str, &str, [F; 2])>,
    );
}
