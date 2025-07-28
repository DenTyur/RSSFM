use crate::config::{C, F, I};
use crate::dim3::{
    gauge::{LenthGauge3D, VelocityGauge3D},
    space::Xspace3D,
};
use crate::macros::check_path;
use crate::traits::{
    field::Field,
    flow::{Flux, SurfaceFlow},
    wave_function::ValueAndSpaceDerivatives,
};
use std::marker::{Send, Sync};

//============================================================================
//                 Поверхность и поток через нее
//============================================================================

/// Плоскость z=border. Поверхность, через которую считается поток.
pub struct Zplane<'a> {
    pub border: F,
    x: &'a Xspace3D,
}

impl<'a> Zplane<'a> {
    pub fn new(border: F, x: &'a Xspace3D) -> Self {
        Self { border, x }
    }
}

impl<'a, G: Flux<3> + Send + Sync> SurfaceFlow<3, G> for Zplane<'a> {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        t: F,
    ) -> C {
        // сейчас считаем, что сетки по всем осям разные
        let ind_xmin: usize = 0;
        let ind_xmax: usize = self.x.n[0];

        let ind_ymin: usize = 0;
        let ind_ymax: usize = self.x.n[1];

        let zmin = self.x.grid[2][[0]];
        let dz = self.x.dx[2];
        let dsz = self.x.dx[0] * self.x.dx[1];
        let ind_zborder = ((self.border - zmin) / dz).round() as usize;

        let compute_z_flow = |ind_border: usize, normale: [F; 3]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix in ind_xmin..ind_xmax {
                for iy in ind_ymin..ind_ymax {
                    let x_point = self.x.grid[0][[ix]];
                    let y_point = self.x.grid[1][[iy]];
                    let z_point = self.x.grid[2][[ind_border]];
                    let point = [x_point, y_point, z_point];
                    let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                    sum += j_flux[0] * normale[0] + j_flux[1] * normale[1] + j_flux[2] * normale[2];
                }
            }
            sum * dsz
        };

        // Суммируем поток через все границы
        let z_right_flow: C = compute_z_flow(ind_zborder, [0.0, 0.0, 1.0]);

        z_right_flow
    }
}

/// Квадрат. Поверхность, через которую считается поток.
pub struct Cube<'a> {
    pub border: F,
    x: &'a Xspace3D,
}

impl<'a> Cube<'a> {
    pub fn new(border: F, x: &'a Xspace3D) -> Self {
        Self { border, x }
    }
}

impl<'a, G: Flux<3> + Send + Sync> SurfaceFlow<3, G> for Cube<'a> {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        t: F,
    ) -> C {
        // сейчас считаем, что сетки по всем осям разные
        let xmin = self.x.grid[0][[0]];
        let dx = self.x.dx[0];
        let dsx = self.x.dx[1] * self.x.dx[2];
        let ind_xmin = ((-self.border - xmin) / dx).round() as usize;
        let ind_xmax = ((self.border - xmin) / dx).round() as usize;

        let ymin = self.x.grid[1][[0]];
        let dy = self.x.dx[1];
        let dsy = self.x.dx[0] * self.x.dx[2];
        let ind_ymin = ((-self.border - ymin) / dy).round() as usize;
        let ind_ymax = ((self.border - ymin) / dy).round() as usize;

        let zmin = self.x.grid[2][[0]];
        let dz = self.x.dx[2];
        let dsz = self.x.dx[0] * self.x.dx[1];
        let ind_zmin = ((-self.border - zmin) / dz).round() as usize;
        let ind_zmax = ((self.border - zmin) / dz).round() as usize;

        let compute_x_flow = |ind_border: usize, normale: [F; 3]| {
            let mut sum: C = C::new(0.0, 0.0);
            for iy in ind_ymin..ind_ymax {
                for iz in ind_zmin..ind_zmax {
                    let x_point = self.x.grid[0][[ind_border]];
                    let y_point = self.x.grid[1][[iy]];
                    let z_point = self.x.grid[2][[iz]];
                    let point = [x_point, y_point, z_point];
                    let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                    sum += j_flux[0] * normale[0] + j_flux[1] * normale[1] + j_flux[2] * normale[2];
                }
            }
            sum * dsx
        };

        let compute_y_flow = |ind_border: usize, normale: [F; 3]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix in ind_xmin..ind_xmax {
                for iz in ind_zmin..ind_zmax {
                    let x_point = self.x.grid[0][[ix]];
                    let y_point = self.x.grid[1][[ind_border]];
                    let z_point = self.x.grid[2][[iz]];
                    let point = [x_point, y_point, z_point];
                    let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                    sum += j_flux[0] * normale[0] + j_flux[1] * normale[1] + j_flux[2] * normale[2];
                }
            }
            sum * dsy
        };

        let compute_z_flow = |ind_border: usize, normale: [F; 3]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix in ind_xmin..ind_xmax {
                for iy in ind_ymin..ind_ymax {
                    let x_point = self.x.grid[0][[ix]];
                    let y_point = self.x.grid[1][[iy]];
                    let z_point = self.x.grid[2][[ind_border]];
                    let point = [x_point, y_point, z_point];
                    let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                    sum += j_flux[0] * normale[0] + j_flux[1] * normale[1] + j_flux[2] * normale[2];
                }
            }
            sum * dsz
        };

        // Суммируем поток через все границы
        let x_left_flow: C = compute_x_flow(ind_xmin, [-1.0, 0.0, 0.0]);
        let x_right_flow: C = compute_x_flow(ind_xmax, [1.0, 0.0, 0.0]);
        let y_left_flow: C = compute_y_flow(ind_ymin, [0.0, -1.0, 0.0]);
        let y_right_flow: C = compute_y_flow(ind_ymax, [0.0, 1.0, 0.0]);
        let z_left_flow: C = compute_z_flow(ind_zmin, [0.0, 0.0, -1.0]);
        let z_right_flow: C = compute_z_flow(ind_zmax, [0.0, 0.0, 1.0]);

        x_left_flow + x_right_flow + y_left_flow + y_right_flow + z_left_flow + z_right_flow
    }
}

//============================================================================
//                 Плотность потока и калибровка
//============================================================================
/// Плотность потока вероятности в калибровке длины
impl<'a, Field3D: Field<3>> Flux<3> for LenthGauge3D<'a, Field3D> {
    fn compute_flux(
        &self,
        x: [F; 3],
        psi1: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        _t: F,
    ) -> [C; 3] {
        let psi1_val = psi1.value(x);
        let psi2_val = psi2.value(x);

        let psi1_derivs = [psi1.deriv(x, 0), psi1.deriv(x, 1), psi1.deriv(x, 2)];
        let psi2_derivs = [psi2.deriv(x, 0), psi2.deriv(x, 1), psi2.deriv(x, 2)];

        let zero_c = C::new(0.0, 0.0);
        let mut j: [C; 3] = [zero_c, zero_c, zero_c];

        for (i, j_elem) in j.iter_mut().enumerate() {
            *j_elem =
                I / 2.0 * (psi2_val * psi1_derivs[i].conj() - psi1_val.conj() * psi2_derivs[i]);
        }
        j
    }
}
/// Плотность потока вероятности в калибровке скорости
impl<'a, Field3D: Field<3>> Flux<3> for VelocityGauge3D<'a, Field3D> {
    fn compute_flux(
        &self,
        x: [F; 3],
        psi1: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        t: F,
    ) -> [C; 3] {
        let vec_pot = self.field.vector_potential(t); // векторный потенциал

        let psi1_val = psi1.value(x);
        let psi2_val = psi2.value(x);

        let psi1_derivs = [psi1.deriv(x, 0), psi1.deriv(x, 1), psi1.deriv(x, 2)];
        let psi2_derivs = [psi2.deriv(x, 0), psi2.deriv(x, 1), psi2.deriv(x, 2)];

        let zero_c = C::new(0.0, 0.0);
        let mut j: [C; 3] = [zero_c, zero_c, zero_c];

        for (i, j_elem) in j.iter_mut().enumerate() {
            *j_elem = psi1_val.conj() * vec_pot[i] * psi2_val
                + I / 2.0 * (psi2_val * psi1_derivs[i].conj() - psi1_val.conj() * psi2_derivs[i]);
        }
        j
    }
}

//============================================================================
//                 Полный поток
//============================================================================
pub struct Flow3D<'a, G: Flux<3> + Send + Sync + Copy, S: SurfaceFlow<3, G> + Sync> {
    gauge: &'a G,
    surface: &'a S,
    pub instance_flow: Vec<C>,
    pub time_instance: Vec<F>,
}

impl<'a, G: Flux<3> + Send + Sync + Copy, S: SurfaceFlow<3, G> + Sync> Flow3D<'a, G, S> {
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
        psi: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        t: F,
    ) -> C {
        self.surface.compute_surface_flow(self.gauge, psi, psi, t)
        // .re // там только действительная часть должна остаться
    }

    pub fn add_instance_flow(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives<3> + Send + Sync),
        t: F,
    ) {
        let instance_flow = self.get_instance_flow(psi, t);
        self.instance_flow.push(instance_flow);
        self.time_instance.push(t);
    }

    pub fn compute_total_flow(&self, dt: F) -> C {
        self.instance_flow.iter().sum::<C>() * dt
    }

    pub fn plot_flow(&self, output_path: &str) {
        use plotters::prelude::*;
        check_path!(output_path);

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
