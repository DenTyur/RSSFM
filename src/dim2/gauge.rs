use super::field::Field2D;
use super::wave_function::WaveFunction2D;
use crate::config::{C, F, I};
use crate::traits::ssfm::GaugedEvolutionSSFM;
use itertools::multizip;
use ndarray::prelude::*;
use rayon::prelude::*;

//================================================================================
//                              VelocityGauge
//================================================================================
/// Калибровка скорости без A^2
#[derive(Clone, Copy)]
pub struct VelocityGauge2D<'a> {
    pub field: &'a Field2D,
}

impl<'a> VelocityGauge2D<'a> {
    pub const DIM: usize = 2;

    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости
impl<'a> GaugedEvolutionSSFM<2> for VelocityGauge2D<'a> {
    type WF = WaveFunction2D;

    fn x_evol_half(
        &self,
        wf: &mut WaveFunction2D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 2]) -> F,
        absorbing_potential: fn(x: [F; 2]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), wf.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        let potential_elem = potential([*x_point, *y_point]);
                        let absorbing_potential_elem = absorbing_potential([*x_point, *y_point]);
                        *psi_elem *=
                            (-I * 0.5 * dt * (potential_elem + absorbing_potential_elem)).exp();
                    },
                );
            });
    }

    fn x_evol(
        &self,
        wf: &mut WaveFunction2D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 2]) -> F,
        absorbing_potential: fn(x: [F; 2]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), wf.x.grid[1].iter())).for_each(
                    |(psi_elem, y_point)| {
                        let potential_elem = potential([*x_point, *y_point]);
                        let absorbing_potential_elem = absorbing_potential([*x_point, *y_point]);
                        *psi_elem *= (-I * dt * (potential_elem + absorbing_potential_elem)).exp();
                    },
                );
            });
    }

    fn p_evol(&self, wf: &mut WaveFunction2D, tcurrent: F, dt: F) {
        let vec_pot = self.field.vec_pot(tcurrent);

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(wf.p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I
                            * dt
                            * (0.5 * px * px + 0.5 * py * py + vec_pot[0] * px + vec_pot[1] * py))
                            .exp();
                    });
            });
    }
}

//================================================================================
//                              LenthGauge
//================================================================================
#[derive(Clone, Copy)]
pub struct LenthGauge2D<'a> {
    pub field: &'a Field2D,
}

impl<'a> LenthGauge2D<'a> {
    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}
