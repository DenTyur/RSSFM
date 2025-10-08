use super::wave_function::WaveFunction1D;
use crate::common::particle::Particle;
use crate::config::{C, F, I};
use crate::traits::{
    field::Field, ssfm::GaugedEvolutionSSFM,
    ssfm_in_imaginary_time::GaugedEvolutionSSFMinImaginaryTime,
};
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

//=========================================================================================
//===================================== SSFM in imaginary time ========================================
//=========================================================================================
/// Эволюция в мнимом времени для SSFM в калибровке скорости для одной одномерной частицы
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFMinImaginaryTime<1> for VelocityGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (0.5 * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn x_evol(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, tcurrent: F, dt: F) {
        let m = particles[0].mass;
        let q = particles[1].charge;
        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (dt * (0.5 / m * px * px + q / m * vec_pot[0] * px)).exp();
            });
    }
}

//=================================================================================
//=====================================SSFM========================================
//=================================================================================
/// Эволюция для SSFM в калибровке скорости для одной одномерной частицы
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<1> for VelocityGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (-I * 0.5 * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn x_evol(
        &self,
        _particles: &[Particle],
        wf: &mut WaveFunction1D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                *psi_elem *= (-I * dt * (potential_elem + absorbing_potential_elem)).exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, tcurrent: F, dt: F) {
        let m = particles[0].mass;
        let q = particles[1].charge;
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
//=====================================SSFM in imaginary time =====================
//=================================================================================
/// Эволюция в мнимом времени для SSFM в калибровке длины для одной одномерной частицы
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFMinImaginaryTime<1> for LenthGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction1D,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                let scalar_potential_energy_elem =
                    particles[0].charge * self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *= (-0.5
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
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
                let scalar_potential_energy_elem =
                    particles[0].charge * self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *= (-dt
                    * (potential_elem + absorbing_potential_elem + scalar_potential_energy_elem))
                    .exp();
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction1D, _tcurrent: F, dt: F) {
        let m = particles[0].mass;
        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-dt * (0.5 / m * px * px)).exp();
            });
    }
}
//=================================================================================
//=====================================SSFM========================================
//=================================================================================
/// Эволюция для SSFM в калибровке длины для одной одномерной частицы
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<1> for LenthGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction1D,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
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
        potential: fn(x: [F; 1]) -> F,
        absorbing_potential: fn(x: [F; 1]) -> C,
    ) {
        multizip((wf.psi.iter_mut(), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, x_point)| {
                let potential_elem = potential([*x_point]);
                let absorbing_potential_elem = absorbing_potential([*x_point]);
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
