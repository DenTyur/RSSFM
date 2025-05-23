use super::field::Field4D;
use super::wave_function::WaveFunction4D;
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
pub struct VelocityGauge4D<'a> {
    pub field: &'a Field4D,
}

impl<'a> VelocityGauge4D<'a> {
    pub const DIM: usize = 4;

    pub fn new(field: &'a Field4D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости
impl<'a> GaugedEvolutionSSFM<4> for VelocityGauge4D<'a> {
    type WF = WaveFunction4D;

    fn x_evol_half(
        &self,
        wf: &mut WaveFunction4D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 4]) -> F,
        absorbing_potential: fn(x: [F; 4]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        let potential_elem =
                                            potential([*x0_point, *x1_point, *x2_point, *x3_point]);
                                        let absorbing_potential_elem = absorbing_potential([
                                            *x0_point, *x1_point, *x2_point, *x3_point,
                                        ]);
                                        *psi_elem *= (-I
                                            * 0.5
                                            * dt
                                            * (potential_elem + absorbing_potential_elem))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn x_evol(
        &self,
        wf: &mut WaveFunction4D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 4]) -> F,
        absorbing_potential: fn(x: [F; 4]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, x0_point)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_2d, x1_point)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[2].iter())).for_each(
                            |(mut psi_1d, x2_point)| {
                                multizip((psi_1d.iter_mut(), wf.x.grid[3].iter())).for_each(
                                    |(psi_elem, x3_point)| {
                                        let potential_elem =
                                            potential([*x0_point, *x1_point, *x2_point, *x3_point]);
                                        let absorbing_potential_elem = absorbing_potential([
                                            *x0_point, *x1_point, *x2_point, *x3_point,
                                        ]);
                                        *psi_elem *=
                                            (-I * dt * (potential_elem + absorbing_potential_elem))
                                                .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn p_evol(&self, wf: &mut WaveFunction4D, tcurrent: F, dt: F) {
        let vec_pot = self.field.vec_pot(tcurrent);

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_3d, p0)| {
                multizip((psi_3d.axis_iter_mut(Axis(0)), wf.p.grid[1].iter())).for_each(
                    |(mut psi_2d, p1)| {
                        multizip((psi_2d.axis_iter_mut(Axis(0)), wf.p.grid[2].iter())).for_each(
                            |(mut psi_1d, p2)| {
                                multizip((psi_1d.iter_mut(), wf.p.grid[3].iter())).for_each(
                                    |(psi_elem, p3)| {
                                        *psi_elem *= (-I
                                            * dt
                                            * (0.5 * p0 * p0
                                                + 0.5 * p1 * p1
                                                + 0.5 * p2 * p2
                                                + 0.5 * p3 * p3
                                                + vec_pot[0] * p0
                                                + vec_pot[1] * p1
                                                + vec_pot[2] * p2
                                                + vec_pot[3] * p3))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }
}

//================================================================================
//                              LenthGauge
//================================================================================
#[derive(Clone, Copy)]
pub struct LenthGauge4D<'a> {
    pub field: &'a Field4D,
}

impl<'a> LenthGauge4D<'a> {
    pub fn new(field: &'a Field4D) -> Self {
        Self { field }
    }
}
