use crate::field;
use crate::flow;
use crate::gauge;
use crate::heatmap;
use crate::logcolormap::plot_heatmap_logscale;
use crate::parameters;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
use flow::{Circle, Flow, Flux, Square, SurfaceFlow};
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
    p: Pspace,
}

impl<'a, G: VolkovGauge + Flux + Send + Sync + Copy, S: SurfaceFlow<G> + Sync> Tsurff<'a, G, S> {
    pub fn new(
        gauge: &'a G,
        surface: &'a S,
        x: &'a Xspace,
        p_true: &Pspace,
        pcut: Option<F>,
    ) -> Self {
        let pcut = match pcut {
            Some(pcut) => pcut,
            None => p_true.grid[0][p_true.n[0] - 1],
        };
        let p_mod = p_true.grid[0]
            .iter()
            .cloned()
            .filter(|&val| val.abs() < pcut)
            .collect::<Array1<_>>();
        let psi_p: Array<C, Ix2> = Array::zeros((p_mod.len(), p_mod.len()));
        let mut p = p_true.clone();
        p.grid[0] = p_mod.clone();
        p.grid[1] = p_mod.clone();
        p.p0 = vec![p_mod[0], p_mod[0]];
        p.n = vec![p_mod.len(), p_mod.len()];
        p.save("arrays_saved/").unwrap();

        Self {
            gauge,
            surface,
            psi_p,
            x,
            p,
        }
    }
    pub fn save(&self, path: &str) {
        let writer = BufWriter::new(File::create(path).unwrap());
        self.psi_p.write_npy(writer).unwrap();
    }

    pub fn plot(&self, path: &str) {
        let mut a: Array2<F> = Array::zeros((self.p.n[0], self.p.n[1]));

        self.psi_p
            .axis_iter(Axis(0))
            .zip(a.axis_iter_mut(Axis(0)))
            .par_bridge()
            .for_each(|(psi_row, mut a_row)| {
                psi_row
                    .iter()
                    .zip(a_row.iter_mut())
                    .for_each(|(psi_elem, a_elem)| {
                        *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
                    })
            });

        let (size_x, size_y, size_colorbar) = (500, 500, 60);
        let (colorbar_min, colorbar_max) = (1e-3, 1e-0);

        heatmap::plot_heatmap(
            &self.p.grid[0],
            &self.p.grid[1],
            &a,
            size_x,
            size_y,
            size_colorbar,
            colorbar_min,
            colorbar_max,
            path,
        )
    }
    pub fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
        let mut a: Array2<F> = Array::zeros((self.p.n[0], self.p.n[1]));

        self.psi_p
            .axis_iter(Axis(0))
            .zip(a.axis_iter_mut(Axis(0)))
            .par_bridge()
            .for_each(|(psi_row, mut a_row)| {
                psi_row
                    .iter()
                    .zip(a_row.iter_mut())
                    .for_each(|(psi_elem, a_elem)| {
                        *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
                    })
            });

        let [colorbar_min, colorbar_max] = colorbar_limits;

        plot_heatmap_logscale(
            &a,
            &self.p.grid[0],
            &self.p.grid[1],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
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
