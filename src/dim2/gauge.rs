use super::wave_function::WaveFunction2D;
use crate::common::particle::Particle;
use crate::config::{C, F, I};
use crate::dim1::gauge::{LenthGauge1D, VelocityGauge1D};
use crate::traits::{field::Field, ssfm::GaugedEvolutionSSFM};
use itertools::multizip;
use ndarray::prelude::*;
use rayon::prelude::*;

//================================================================================
//                              VelocityGauge
//================================================================================
/// Калибровка скорости без A^2
#[derive(Clone, Copy)]
pub struct VelocityGauge2D<'a, Field2D: Field<2>> {
    pub field: &'a Field2D,
}

impl<'a, Field2D: Field<2>> VelocityGauge2D<'a, Field2D> {
    pub const DIM: usize = 2;

    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке скорости для одной двумерной частицы
impl<'a, Field2D: Field<2>> GaugedEvolutionSSFM<2> for VelocityGauge2D<'a, Field2D> {
    type WF = WaveFunction2D;

    fn x_evol_half(
        &self,
        _particle: &[Particle],
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
        _particles: &[Particle],
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

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction2D, tcurrent: F, dt: F) {
        let m = particles[0].mass;
        let q = particles[0].charge;
        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(wf.p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I
                            * dt
                            * (0.5 / m * px * px
                                + 0.5 / m * py * py
                                + q / m * vec_pot[0] * px
                                + q / m * vec_pot[1] * py))
                            .exp();
                    });
            });
    }
}

/// Эволюция для SSFM в калибровке скорости для двух одномерных частиц
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<2> for VelocityGauge1D<'a, Field1D> {
    type WF = WaveFunction2D;

    fn x_evol_half(
        &self,
        _particle: &[Particle],
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
        _particles: &[Particle],
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

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction2D, tcurrent: F, dt: F) {
        let m0 = particles[0].mass;
        let q0 = particles[0].charge;
        let m1 = particles[1].mass;
        let q1 = particles[1].charge;

        let vec_pot = self.field.vector_potential(tcurrent);

        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(wf.p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I
                            * dt
                            * (0.5 / m0 * px * px
                                + 0.5 / m1 * py * py
                                + q0 / m0 * vec_pot[0] * px
                                + q1 / m1 * vec_pot[0] * py))
                            .exp();
                    });
            });
    }
}

//================================================================================
//                              LenthGauge
//================================================================================
#[derive(Clone, Copy)]
pub struct LenthGauge2D<'a, Field2D: Field<2>> {
    pub field: &'a Field2D,
}

impl<'a, Field2D: Field<2>> LenthGauge2D<'a, Field2D> {
    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}

//=====================================SSFM========================================
/// Эволюция для SSFM в калибровке длины для одной двумерной частицы Поле 2D
impl<'a, Field2D: Field<2>> GaugedEvolutionSSFM<2> for LenthGauge2D<'a, Field2D> {
    type WF = WaveFunction2D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction2D,
        tcurrent: F,
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
                        let scalar_potential_energy_elem = particles[0].charge
                            * self.field.scalar_potential([*x_point, *y_point], tcurrent);
                        *psi_elem *= (-I
                            * 0.5
                            * dt
                            * (potential_elem
                                + absorbing_potential_elem
                                + scalar_potential_energy_elem))
                            .exp();
                    },
                );
            });
    }

    fn x_evol(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction2D,
        tcurrent: F,
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
                        let scalar_potential_energy_elem = particles[0].charge
                            * self.field.scalar_potential([*x_point, *y_point], tcurrent);
                        *psi_elem *= (-I
                            * dt
                            * (potential_elem
                                + absorbing_potential_elem
                                + scalar_potential_energy_elem))
                            .exp();
                    },
                );
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction2D, _tcurrent: F, dt: F) {
        let m = particles[0].mass;
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(wf.p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I * dt * (0.5 / m * px * px + 0.5 / m * py * py)).exp();
                    });
            });
    }
}

/// Эволюция для SSFM в калибровке длины для двух одномерных частиц Поле 1D
impl<'a, Field1D: Field<1>> GaugedEvolutionSSFM<2> for LenthGauge1D<'a, Field1D> {
    type WF = WaveFunction2D;

    fn x_evol_half(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction2D,
        tcurrent: F,
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
                        let scalar_potential_energy_elem = particles[0].charge
                            * self.field.scalar_potential([*x_point], tcurrent)
                            + particles[1].charge
                                * self.field.scalar_potential([*y_point], tcurrent);
                        *psi_elem *= (-I
                            * 0.5
                            * dt
                            * (potential_elem
                                + absorbing_potential_elem
                                + scalar_potential_energy_elem))
                            .exp();
                    },
                );
            });
    }

    fn x_evol(
        &self,
        particles: &[Particle],
        wf: &mut WaveFunction2D,
        tcurrent: F,
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
                        let scalar_potential_energy_elem = particles[0].charge
                            * self.field.scalar_potential([*x_point], tcurrent)
                            + particles[1].charge
                                * self.field.scalar_potential([*y_point], tcurrent);
                        *psi_elem *= (-I
                            * dt
                            * (potential_elem
                                + absorbing_potential_elem
                                + scalar_potential_energy_elem))
                            .exp();
                    },
                );
            });
    }

    fn p_evol(&self, particles: &[Particle], wf: &mut WaveFunction2D, _tcurrent: F, dt: F) {
        let m0 = particles[0].mass;
        let m1 = particles[1].mass;
        multizip((wf.psi.axis_iter_mut(Axis(0)), wf.p.grid[0].iter()))
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(wf.p.grid[1].iter())
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I * dt * (0.5 / m0 * px * px + 0.5 / m1 * py * py)).exp();
                    });
            });
    }
}
