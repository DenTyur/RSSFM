use super::field::Field4D;
use super::wave_function::WaveFunction4D;
use crate::common::particle::Particle;
use crate::config::{C, F, I};
use crate::dim2::gauge::{LenthGauge2D, VelocityGauge2D};
use crate::traits::{field::Field, ssfm::GaugedEvolutionSSFM};
use itertools::multizip;
use ndarray::prelude::*;
use rayon::prelude::*;

//================================================================================
//                              VelocityGauge
//================================================================================
/// Калибровка скорости без A^2
#[derive(Clone, Copy)]
pub struct VelocityGauge4D<'a, Field4D: Field<4>> {
    pub field: &'a Field4D,
}

impl<'a, Field4D: Field<4>> VelocityGauge4D<'a, Field4D> {
    pub const DIM: usize = 4;

    pub fn new(field: &'a Field4D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости для двух четырехмерных частиц Поле 2D
impl<'a, Field2D: Field<2>> GaugedEvolutionSSFM<4> for VelocityGauge2D<'a, Field2D> {
    type WF = WaveFunction4D;

    fn x_evol_half(
        &self,
        _particles: &[Particle],
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
        _particles: &[Particle],
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

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction4D, tcurrent: F, dt: F) {
        let m = particles[0].mass;
        let q = particles[0].charge;
        let vec_pot = self.field.vector_potential(tcurrent);

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
                                            * (0.5 / m * p0 * p0
                                                + 0.5 / m * p1 * p1
                                                + 0.5 / m * p2 * p2
                                                + 0.5 / m * p3 * p3
                                                + q / m * vec_pot[0] * p0
                                                + q / m * vec_pot[1] * p1
                                                + q / m * vec_pot[2] * p2
                                                + q / m * vec_pot[3] * p3))
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
pub struct LenthGauge4D<'a, Field4D: Field<4>> {
    pub field: &'a Field4D,
}

impl<'a, Field4D: Field<4>> LenthGauge4D<'a, Field4D> {
    pub fn new(field: &'a Field4D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке длины для двух двумерных частиц Поле 2D
impl<'a, Field2D: Field<2>> GaugedEvolutionSSFM<4> for LenthGauge2D<'a, Field2D> {
    type WF = WaveFunction4D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction4D,
        tcurrent: F,
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
                                        let scalar_potential_energy_elem = particles[0].charge
                                            * self
                                                .field
                                                .scalar_potential([*x0_point, *x1_point], tcurrent)
                                            + particles[1].charge
                                                * self.field.scalar_potential(
                                                    [*x2_point, *x3_point],
                                                    tcurrent,
                                                );
                                        *psi_elem *= (-I
                                            * 0.5
                                            * dt
                                            * (potential_elem
                                                + absorbing_potential_elem
                                                + scalar_potential_energy_elem))
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
        particles: &[Particle],
        wf: &mut WaveFunction4D,
        tcurrent: F,
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
                                        let scalar_potential_energy_elem = particles[0].charge
                                            * self
                                                .field
                                                .scalar_potential([*x0_point, *x1_point], tcurrent)
                                            + particles[1].charge
                                                * self.field.scalar_potential(
                                                    [*x2_point, *x3_point],
                                                    tcurrent,
                                                );
                                        *psi_elem *= (-I
                                            * dt
                                            * (potential_elem
                                                + absorbing_potential_elem
                                                + scalar_potential_energy_elem))
                                            .exp();
                                    },
                                );
                            },
                        );
                    },
                );
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction4D, tcurrent: F, dt: F) {
        let m0 = particles[0].mass;
        let q0 = particles[0].charge;
        let m1 = particles[1].mass;
        let q1 = particles[1].charge;
        let vec_pot = self.field.vector_potential(tcurrent);

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
                                            * (0.5 / m0 * p0 * p0
                                                + 0.5 / m0 * p1 * p1
                                                + 0.5 / m1 * p2 * p2
                                                + 0.5 / m1 * p3 * p3
                                                + q0 / m0 * vec_pot[0] * p0
                                                + q0 / m0 * vec_pot[1] * p1
                                                + q1 / m1 * vec_pot[2] * p2
                                                + q1 / m1 * vec_pot[3] * p3))
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
