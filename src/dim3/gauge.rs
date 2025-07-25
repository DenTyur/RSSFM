use super::wave_function::WaveFunction3D;
use crate::config::{C, F, I};
use crate::traits::{field::Field, ssfm::GaugedEvolutionSSFM};
use itertools::multizip;
use ndarray::prelude::*;
use rayon::prelude::*;

//================================================================================
//                              VelocityGauge
//================================================================================
/// Калибровка скорости без A^2
#[derive(Clone, Copy)]
pub struct VelocityGauge3D<'a, Field3D: Field<3>> {
    pub field: &'a Field3D,
}

impl<'a, Field3D: Field<3>> VelocityGauge3D<'a, Field3D> {
    pub const DIM: usize = 3;

    pub fn new(field: &'a Field3D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости
impl<'a, Field3D: Field<3>> GaugedEvolutionSSFM<3> for VelocityGauge3D<'a, Field3D> {
    type WF = WaveFunction3D;

    fn x_evol_half(
        &self,
        wf: &mut WaveFunction3D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 3]) -> F,
        absorbing_potential: fn(x: [F; 3]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                let potential_elem = potential([*x0_point, *x1_point, *x2_point]);
                                let absorbing_potential_elem =
                                    absorbing_potential([*x0_point, *x1_point, *x2_point]);
                                *psi_elem *=
                                    (-I * 0.5 * dt * (potential_elem + absorbing_potential_elem))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }

    fn x_evol(
        &self,
        wf: &mut WaveFunction3D,
        _tcurrent: F,
        dt: F,
        potential: fn(x: [F; 3]) -> F,
        absorbing_potential: fn(x: [F; 3]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                let potential_elem = potential([*x0_point, *x1_point, *x2_point]);
                                let absorbing_potential_elem =
                                    absorbing_potential([*x0_point, *x1_point, *x2_point]);
                                *psi_elem *=
                                    (-I * dt * (potential_elem + absorbing_potential_elem)).exp();
                            },
                        );
                    },
                );
            });
    }

    fn p_evol(&self, wf: &mut WaveFunction3D, tcurrent: F, dt: F) {
        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, p0)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.p.grid[1].iter())).for_each(
                    |(mut psi_1d, p1)| {
                        multizip((psi_1d.iter_mut(), wf.p.grid[2].iter())).for_each(
                            |(psi_elem, p2)| {
                                *psi_elem *= (-I
                                    * dt
                                    * (0.5 * p0 * p0
                                        + 0.5 * p1 * p1
                                        + 0.5 * p2 * p2
                                        + vec_pot[0] * p0
                                        + vec_pot[1] * p1
                                        + vec_pot[2] * p2))
                                    .exp();
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
pub struct LenthGauge3D<'a, Field3D: Field<3>> {
    pub field: &'a Field3D,
}

impl<'a, Field3D: Field<3>> LenthGauge3D<'a, Field3D> {
    pub fn new(field: &'a Field3D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке длины
impl<'a, Field3D: Field<3>> GaugedEvolutionSSFM<3> for LenthGauge3D<'a, Field3D> {
    type WF = WaveFunction3D;

    fn x_evol_half(
        &self,
        wf: &mut WaveFunction3D,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; 3]) -> F,
        absorbing_potential: fn(x: [F; 3]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                let potential_elem = potential([*x0_point, *x1_point, *x2_point]);
                                let absorbing_potential_elem =
                                    absorbing_potential([*x0_point, *x1_point, *x2_point]);
                                let scalar_potential_elem = self
                                    .field
                                    .scalar_potential([*x0_point, *x1_point, *x2_point], tcurrent);
                                *psi_elem *= (-I
                                    * 0.5
                                    * dt
                                    * (potential_elem + absorbing_potential_elem
                                        - scalar_potential_elem))
                                    .exp();
                            },
                        );
                    },
                );
            });
    }

    fn x_evol(
        &self,
        wf: &mut WaveFunction3D,
        tcurrent: F,
        dt: F,
        potential: fn(x: [F; 3]) -> F,
        absorbing_potential: fn(x: [F; 3]) -> C,
    ) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.x.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, x0_point)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.x.grid[1].iter())).for_each(
                    |(mut psi_1d, x1_point)| {
                        multizip((psi_1d.iter_mut(), wf.x.grid[2].iter())).for_each(
                            |(psi_elem, x2_point)| {
                                let potential_elem = potential([*x0_point, *x1_point, *x2_point]);
                                let absorbing_potential_elem =
                                    absorbing_potential([*x0_point, *x1_point, *x2_point]);
                                let scalar_potential_elem = self
                                    .field
                                    .scalar_potential([*x0_point, *x1_point, *x2_point], tcurrent);
                                *psi_elem *= (-I
                                    * dt
                                    * (potential_elem + absorbing_potential_elem
                                        - scalar_potential_elem))
                                    .exp();
                            },
                        );
                    },
                );
            });
    }

    fn p_evol(&self, wf: &mut WaveFunction3D, tcurrent: F, dt: F) {
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_2d, p0)| {
                multizip((psi_2d.axis_iter_mut(Axis(0)), wf.p.grid[1].iter())).for_each(
                    |(mut psi_1d, p1)| {
                        multizip((psi_1d.iter_mut(), wf.p.grid[2].iter())).for_each(
                            |(psi_elem, p2)| {
                                *psi_elem *=
                                    (-I * dt * (0.5 * p0 * p0 + 0.5 * p1 * p1 + 0.5 * p2 * p2))
                                        .exp();
                            },
                        );
                    },
                );
            });
    }
}
