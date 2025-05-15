use super::space::Pspace2D;
use super::volkov::Volkov2D;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::macros::check_path;
use crate::traits::tsurff::Tsurff;
use crate::traits::{
    flow::{Flux, SurfaceFlow},
    volkov::VolkovGauge,
    wave_function::ValueAndSpaceDerivatives,
};
use crate::utils::{heatmap, logcolormap};
use ndarray::prelude::*;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::marker::{Send, Sync};

pub struct Tsurff2D<'a, G, S>
where
    G: VolkovGauge + Flux<2> + Send + Sync + Copy,
    S: SurfaceFlow<2, G> + Sync,
{
    gauge: &'a G,
    surface: &'a S,
    pub psi_p: Array<C, Ix2>,
    pub p: Pspace2D,
}

impl<'a, G, S> Tsurff<'a, 2, G, S> for Tsurff2D<'a, G, S>
where
    G: VolkovGauge + Flux<2> + Send + Sync + Copy,
    S: SurfaceFlow<2, G> + Sync,
{
    type Pspace = Pspace2D;

    fn new(gauge: &'a G, surface: &'a S, p_true: &Pspace2D, pcut: Option<F>) -> Self {
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
        p.p0 = [p_mod[0], p_mod[0]];
        p.n = [p_mod.len(), p_mod.len()];

        Self {
            gauge,
            surface,
            psi_p,
            p,
        }
    }

    fn save_as_npy(&self, path: &str) {
        check_path!(path);
        let writer = BufWriter::new(File::create(path).unwrap());
        self.psi_p.write_npy(writer).unwrap();
    }

    fn plot(&self, path: &str, colorbar_limits: [F; 2]) {
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

        check_path!(path);
        heatmap::plot_heatmap(
            &self.p.grid[0],
            &self.p.grid[1],
            &a,
            size_x,
            size_y,
            size_colorbar,
            colorbar_limits[0],
            colorbar_limits[1],
            path,
        )
    }
    fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
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

        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &a,
            &self.p.grid[0],
            &self.p.grid[1],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }

    fn time_integration_step(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives<2> + Send + Sync),
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
                        // Инициализируем Волковские функции с соответствующими импульсами
                        let volkov = Volkov2D::new(self.gauge, [*px, *py], t.current);
                        *psi_p_elem += self
                            .surface
                            .compute_surface_flow(self.gauge, &volkov, psi, t.current)
                            * t.t_step();
                    })
            });
    }
}
