use super::wave_function::WaveFunction1D;
use crate::common::particle::Particle;
use crate::config::{C, F, I};
use crate::traits::{field::Field, ssfm::GaugedEvolutionSSFM};
use itertools::multizip;
use rayon::prelude::*;

//================================================================================
//                              VelocityGauge
//================================================================================
/// Калибровка скорости без A^2
#[derive(Clone, Copy)]
pub struct VelocityGauge1D<'a, Field1D: Field<1>> {
    pub field: &'a Field1D,
}

impl<'a, Field1D: Field<1>> VelocityGauge1D<'a, Field1D> {
    pub const DIM: usize = 1;

    pub fn new(field: &'a Field1D) -> Self {
        Self { field }
    }
}

//=================================================================================
//=====================================SSFM========================================
//=================================================================================
/// Эволюция для SSFM в калибровке скорости для одной одномерной частицы
impl<'a, Field1D, AP, AB> GaugedEvolutionSSFM<1, AP, AB> for VelocityGauge1D<'a, Field1D>
where
    Field1D: Field<1>,
    AP: Fn([F; 1]) -> F + Send + Sync,
    AB: Fn([F; 1]) -> C + Send + Sync,
{
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: &AP,           // Изменено на &AP
        absorbing_potential: &AB, // Изменено на &AB
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]); // Вызов через обобщенный тип
                let absorbing_potential_elem = absorbing_potential([*x_point]); // Вызов через обобщенный тип
                *psi_elem *= (-I * 0.5 * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn x_evol(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: &AP,           // Изменено на &AP
        absorbing_potential: &AB, // Изменено на &AB
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]); // Вызов через обобщенный тип
                let absorbing_potential_elem = absorbing_potential([*x_point]); // Вызов через обобщенный тип
                *psi_elem *= (-I * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, tcurrent: F, dt: F) {
        // Исправлен индекс - должна быть particles[0], так как передается одна частица
        let m = particles[0].mass;
        let q = particles[0].charge; // Изменено с [1] на [0]
        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-I * dt * (0.5 / m * px * px + q / m * vec_pot[0] * px)).exp();
            });
    }
}

//================================================================================
//                              LenthGauge
//================================================================================
#[derive(Clone, Copy)]
pub struct LenthGauge1D<'a, Field1D: Field<1>> {
    pub field: &'a Field1D,
}

impl<'a, Field1D: Field<1>> LenthGauge1D<'a, Field1D> {
    pub fn new(field: &'a Field1D) -> Self {
        Self { field }
    }
}

//=================================================================================
//=====================================SSFM========================================
//=================================================================================
/// Эволюция для SSFM в калибровке длины для одной одномерной частицы
impl<'a, Field1D, AP, AB> GaugedEvolutionSSFM<1, AP, AB> for LenthGauge1D<'a, Field1D>
where
    Field1D: Field<1>,
    AP: Fn([F; 1]) -> F + Send + Sync,
    AB: Fn([F; 1]) -> C + Send + Sync,
{
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction1D,
        tcurrent: F,
        dt: F,
        potential: &AP,           // Изменено на &AP
        absorbing_potential: &AB, // Изменено на &AB
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]); // Вызов через обобщенный тип
                let absorbing_potential_elem = absorbing_potential([*x_point]); // Вызов через обобщенный тип
                let scalar_potential_energy_elem =
                    particles[0].charge * self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *= (-I
                    * 0.5
                    * dt
                    * (potential_elem + absorbing_potential_elem + scalar_potential_energy_elem))
                    .exp();
            });
    }

    fn x_evol(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction1D,
        tcurrent: F,
        dt: F,
        potential: &AP,           // Изменено на &AP
        absorbing_potential: &AB, // Изменено на &AB
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]); // Вызов через обобщенный тип
                let absorbing_potential_elem = absorbing_potential([*x_point]); // Вызов через обобщенный тип
                let scalar_potential_energy_elem =
                    particles[0].charge * self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *= (-I
                    * dt
                    * (potential_elem + absorbing_potential_elem + scalar_potential_energy_elem))
                    .exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, _tcurrent: F, dt: F) {
        let m = particles[0].mass;
        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-I * dt * (0.5 / m * px * px)).exp();
            });
    }
}
