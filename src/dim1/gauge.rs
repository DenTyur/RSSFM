use super::wave_function::WaveFunction1D;
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

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<1> for VelocityGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
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

    fn p_evol(&self, wf: &mut WaveFunction1D, tcurrent: F, dt: F) {
        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-I * dt * (0.5 * px * px + vec_pot[0] * px)).exp();
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

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке длины
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<1> for LenthGauge1D<'a, Field1D> {
    type WF = WaveFunction1D;

    fn x_evol_half(
        &self,
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
                let scalar_potential_elem = self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *= (-I
                    * 0.5
                    * dt
                    * (potential_elem + absorbing_potential_elem - scalar_potential_elem))
                    .exp();
            });
    }

    fn x_evol(
        &self,
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
                let scalar_potential_elem = self.field.scalar_potential([*x_point], tcurrent);
                *psi_elem *=
                    (-I * dt * (potential_elem + absorbing_potential_elem - scalar_potential_elem))
                        .exp();
            });
    }

    fn p_evol(&self, wf: &mut WaveFunction1D, _tcurrent: F, dt: F) {
        multizip((wf.psi.iter_mut(), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(psi_elem, px)| {
                *psi_elem *= (-I * dt * (0.5 * px * px)).exp();
            });
    }
}
