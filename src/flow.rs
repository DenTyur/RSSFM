use crate::field;
use crate::gauge;
use crate::parameters;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
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

pub struct Circle {
    r: F,
    n_points: usize,
    dphi: F,
    ds: F,
}

impl Circle {
    pub fn new(r: F, n_points: usize) -> Self {
        let dphi = 2.0 * PI / n_points as F;
        let ds = r * dphi;

        Self {
            r,
            n_points,
            dphi,
            ds,
        }
    }

    pub fn normale_in_point(&self, i: usize) -> [F; 2] {
        let phi = self.get_phi_in_point(i);
        [phi.cos(), phi.sin()]
    }

    pub fn get_phi_in_point(&self, i: usize) -> F {
        i as F * self.dphi
    }

    pub fn coords_in_point(&self, i: usize) -> [F; 2] {
        let phi = self.get_phi_in_point(i);
        [self.r * phi.cos(), self.r * phi.sin()]
    }
}

pub struct Square<'a> {
    border: F,
    x: &'a Xspace,
}

impl<'a> Square<'a> {
    pub fn new(border: F, x: &'a Xspace) -> Self {
        Self { border, x }
    }
}

pub trait SurfaceFlow<G: Flux + Send + Sync> {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> C;
}

impl<'a, G: Flux + Send + Sync> SurfaceFlow<G> for Square<'a> {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> C {
        // сейчас считаем, что сетки по x и y одинаковые
        let xmin = self.x.grid[0][[0]];
        let ds = self.x.dx[0];
        let ind_min = ((-self.border - xmin) / ds).round() as usize;
        let ind_max = ((self.border - xmin) / ds).round() as usize;

        let compute_x_flow = |ind_border: usize, normale: [F; 2]| {
            (ind_min..ind_max)
                .map(|i| {
                    let x_point = self.x.grid[0][[i]];
                    let y_point = self.x.grid[1][[ind_border]];
                    let j_flux = gauge.compute_flux([x_point, y_point], psi1, psi2, t);
                    j_flux[0] * normale[0] + j_flux[1] * normale[1]
                })
                .sum::<C>()
                * ds
        };

        let compute_y_flow = |ind_border: usize, normale: [F; 2]| {
            (ind_min..ind_max)
                .map(|i| {
                    let y_point = self.x.grid[1][[i]];
                    let x_point = self.x.grid[0][[ind_border]];
                    let j_flux = gauge.compute_flux([x_point, y_point], psi1, psi2, t);
                    j_flux[0] * normale[0] + j_flux[1] * normale[1]
                })
                .sum::<C>()
                * ds
        };

        // Суммируем поток через все границы
        let left_flow: C = compute_x_flow(ind_min, [0.0, -1.0]);
        let right_flow: C = compute_x_flow(ind_max, [0.0, 1.0]);
        let bottom_flow: C = compute_y_flow(ind_min, [-1.0, 0.0]);
        let top_flow: C = compute_y_flow(ind_max, [1.0, 0.0]);

        left_flow + right_flow + bottom_flow + top_flow
    }
}

impl<G: Flux + Send + Sync> SurfaceFlow<G> for Circle {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> C {
        // Интегрируем поток через поверхность:
        (0..self.n_points)
            .into_par_iter()
            .map(|i| {
                // вычисление точек поверхности и ее нормали в прямоугольных координатах
                let x = self.coords_in_point(i);
                let normale = self.normale_in_point(i);

                let j_flux = gauge.compute_flux(x, psi1, psi2, t);

                // проекция потока на нормаль к поверхности
                j_flux[0] * normale[0] + j_flux[1] * normale[1]
            })
            .sum::<C>()
            * self.ds
    }
}

pub trait Flux {
    // вычисляет вектор потока (вероятности или невероятности)
    fn compute_flux(
        &self,
        x: [F; 2],
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> [C; 2];
}

impl<'a> Flux for VelocityGauge<'a> {
    fn compute_flux(
        &self,
        x: [F; 2],
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> [C; 2] {
        let vec_pot = self.field.vec_pot(t); // векторный потенциал

        let psi1_val = psi1.value(x);
        let psi2_val = psi2.value(x);

        let psi1_derivs = psi1.deriv(x);
        let psi2_derivs = psi2.deriv(x);

        let j0 = psi1_val.conj() * vec_pot[0] * psi2_val
            + I / 2.0 * (psi2_val * psi1_derivs[0].conj() - psi1_val.conj() * psi2_derivs[0]);

        let j1 = psi1_val.conj() * vec_pot[1] * psi2_val
            + I / 2.0 * (psi2_val * psi1_derivs[1].conj() - psi1_val.conj() * psi2_derivs[1]);

        [j0, j1]
    }
}

impl<'a> Flux for LenthGauge<'a> {
    fn compute_flux(
        &self,
        x: [F; 2],
        psi1: &(impl ValueAndSpaceDerivatives + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives + Send + Sync),
        _t: F,
    ) -> [C; 2] {
        let psi1_val = psi1.value(x);
        let psi2_val = psi2.value(x);

        let psi1_derivs = psi1.deriv(x);
        let psi2_derivs = psi2.deriv(x);

        let j0 = I / 2.0 * (psi2_val * psi1_derivs[0].conj() - psi1_val.conj() * psi2_derivs[0]);

        let j1 = I / 2.0 * (psi2_val * psi1_derivs[1].conj() - psi1_val.conj() * psi2_derivs[1]);

        [j0, j1]
    }
}
pub struct Flow<'a, G: VolkovGauge + Flux + Send + Sync + Copy, S: SurfaceFlow<G> + Sync> {
    gauge: &'a G,
    surface: &'a S,
    instance_flow: Vec<C>,
    time_instance: Vec<F>,
}

impl<'a, G: VolkovGauge + Flux + Send + Sync + Copy, S: SurfaceFlow<G> + Sync> Flow<'a, G, S> {
    pub fn new(gauge: &'a G, surface: &'a S) -> Self {
        let instance_flow: Vec<C> = Vec::new();
        let time_instance: Vec<F> = Vec::new();
        Self {
            gauge,
            surface,
            instance_flow,
            time_instance,
        }
    }

    pub fn get_instance_flow(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives + Send + Sync),
        t: F,
    ) -> C {
        // Кайф!
        self.surface.compute_surface_flow(self.gauge, psi, psi, t)
        // .re // там только действительная часть должна остаться
    }

    pub fn add_instance_flow(&mut self, psi: &(impl ValueAndSpaceDerivatives + Send + Sync), t: F) {
        let instance_flow = self.get_instance_flow(psi, t);
        self.instance_flow.push(instance_flow);
        self.time_instance.push(t);
    }

    pub fn compute_total_flow(&self, dt: F) -> C {
        self.instance_flow.iter().sum::<C>() * dt
    }

    pub fn plot_flow(&self, output_path: &str) {
        use plotters::prelude::*;

        // Создаём область для графика
        let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Автоматически вычисляем диапазоны осей
        let x_range = self
            .time_instance
            .iter()
            .fold(F::INFINITY..F::NEG_INFINITY, |range, &val| {
                val.min(range.start)..val.max(range.end)
            });

        let y_range = self
            .instance_flow
            .iter()
            .fold(F::INFINITY..F::NEG_INFINITY, |range, &val| {
                val.re.min(range.start)..val.re.max(range.end)
            });

        // Добавляем 10% отличия по краям для лучшего отображения
        let x_padding = (x_range.end - x_range.start) * 0.1;
        let y_padding = (y_range.end - y_range.start) * 0.1;

        let x_range = (x_range.start - x_padding)..(x_range.end + x_padding);
        let y_range = (y_range.start - y_padding)..(y_range.end + y_padding);

        // Создаём график
        let mut chart = ChartBuilder::on(&root)
            .caption("Поток вероятности", ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(x_range, y_range)
            .unwrap();

        // Настраиваем сетку
        chart
            .configure_mesh()
            .x_desc("t")
            .y_desc("flow")
            .draw()
            .unwrap();

        // Real Flow
        // Рисуем линию графика
        chart
            .draw_series(LineSeries::new(
                self.time_instance
                    .iter()
                    .zip(self.instance_flow.iter())
                    .map(|(&x, &y)| (x, y.re)),
                RED.stroke_width(2),
            ))
            .unwrap();

        // Добавляем маркеры точек
        chart
            .draw_series(
                self.time_instance
                    .iter()
                    .zip(self.instance_flow.iter())
                    .map(|(&x, &y)| Circle::new((x, y.re), 3, RED.filled())),
            )
            .unwrap();

        // Imag Flow
        // Рисуем линию графика
        chart
            .draw_series(LineSeries::new(
                self.time_instance
                    .iter()
                    .zip(self.instance_flow.iter())
                    .map(|(&x, &y)| (x, y.im)),
                BLUE.stroke_width(2),
            ))
            .unwrap();

        // Добавляем маркеры точек
        chart
            .draw_series(
                self.time_instance
                    .iter()
                    .zip(self.instance_flow.iter())
                    .map(|(&x, &y)| Circle::new((x, y.im), 3, BLUE.filled())),
            )
            .unwrap();
    }
}
