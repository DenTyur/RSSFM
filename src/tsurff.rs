use crate::field;
use crate::flow;
use crate::gauge;
use crate::parameters;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
use flow::{Flux, SurfaceFlow};
use gauge::{LenthGauge, VelocityGauge};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::marker::{Send, Sync};
use volkov::{Volkov, VolkovGauge};
use wave_function::{ValueAndSpaceDerivatives, WaveFunction};

type F = f32;
type C = Complex<F>;
const I: C = Complex::I;

pub struct Tsurff<'a, G: VolkovGauge + Flux + Send + Sync + Copy, S: SurfaceFlow<G> + Sync> {
    gauge: &'a G,
    surface: &'a S,
    pub psi_p: Array<C, Ix2>,
    x: &'a Xspace,
    p: &'a Pspace,
}

impl<'a, G: VolkovGauge + Flux + Send + Sync + Copy, S: SurfaceFlow<G> + Sync> Tsurff<'a, G, S> {
    pub fn new(gauge: &'a G, surface: &'a S, x: &'a Xspace, p: &'a Pspace) -> Self {
        let psi_p: Array<C, Ix2> = Array::zeros((p.n[0], p.n[1]));

        Self {
            gauge,
            surface,
            psi_p,
            x,
            p,
        }
    }

    pub fn time_integration_step(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: &Tspace,
    ) {
        self.psi_p
            .axis_iter_mut(Axis(0))
            .zip(self.p.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_p_row, px)| {
                psi_p_row
                    .iter_mut()
                    .zip(self.p.grid[1].iter())
                    .for_each(|(psi_p_elem, py)| {
                        // итераторы: 0-ось psi_p сзипована с px
                        //            1-ось psi_p сзипована с py
                        // Инициализируем Волковские функции с соответствующими импульсами
                        let volkov = Volkov::new(self.gauge, [*px, *py], t.current);
                        *psi_p_elem += self
                            .surface
                            .compute_surface_flow(self.gauge, &volkov, psi, t.current)
                            * t.t_step();
                    })
            });
    }
}
