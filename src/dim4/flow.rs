use crate::config::{C, F, I};
use crate::dim4::{gauge::VelocityGauge4D, space::Xspace4D};
use crate::macros::check_path;
use crate::traits::{
    flow::{Flux, SurfaceFlow},
    wave_function::ValueAndSpaceDerivatives,
};
use std::marker::{Send, Sync};

//============================================================================
//                 Поверхность и поток через нее
//============================================================================

/// Квадрат. Поверхность, через которую считается поток.
pub struct Hypercube<'a> {
    pub border: F,
    x: &'a Xspace4D,
}

impl<'a> Hypercube<'a> {
    pub fn new(border: F, x: &'a Xspace4D) -> Self {
        Self { border, x }
    }
}

impl<'a, G: Flux<4> + Send + Sync> SurfaceFlow<4, G> for Hypercube<'a> {
    fn compute_surface_flow(
        &self,
        gauge: &G,
        psi1: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
        t: F,
    ) -> C {
        // сейчас считаем, что сетки по всем осям одинаковые
        let xmin = self.x.grid[0][[0]];
        let ds = self.x.dx[0] * self.x.dx[1] * self.x.dx[2];
        let dx = self.x.dx[0];
        let ind_min = ((-self.border - xmin) / dx).round() as usize;
        let ind_max = ((self.border - xmin) / dx).round() as usize;
        // println!("indmin = {}, indmax = {}", ind_min, ind_max);

        let compute_x1_flow = |ind_border: usize, normale: [F; 4]| {
            let mut sum: C = C::new(0.0, 0.0);
            for iy1 in ind_min..ind_max {
                for ix2 in ind_min..ind_max {
                    for iy2 in ind_min..ind_max {
                        let x1_point = self.x.grid[0][[ind_border]];
                        let y1_point = self.x.grid[1][[iy1]];
                        let x2_point = self.x.grid[2][[ix2]];
                        let y2_point = self.x.grid[3][[iy2]];
                        let point = [x1_point, y1_point, x2_point, y2_point];
                        let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                        sum += j_flux[0] * normale[0]
                            + j_flux[1] * normale[1]
                            + j_flux[2] * normale[2]
                            + j_flux[3] * normale[3];
                    }
                }
            }
            sum * ds
        };

        let compute_y1_flow = |ind_border: usize, normale: [F; 4]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix1 in ind_min..ind_max {
                for ix2 in ind_min..ind_max {
                    for iy2 in ind_min..ind_max {
                        let x1_point = self.x.grid[0][[ix1]];
                        let y1_point = self.x.grid[1][[ind_border]];
                        let x2_point = self.x.grid[2][[ix2]];
                        let y2_point = self.x.grid[3][[iy2]];
                        let point = [x1_point, y1_point, x2_point, y2_point];
                        let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                        sum += j_flux[0] * normale[0]
                            + j_flux[1] * normale[1]
                            + j_flux[2] * normale[2]
                            + j_flux[3] * normale[3];
                    }
                }
            }
            sum * ds
        };

        let compute_x2_flow = |ind_border: usize, normale: [F; 4]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix1 in ind_min..ind_max {
                for iy1 in ind_min..ind_max {
                    for iy2 in ind_min..ind_max {
                        let x1_point = self.x.grid[0][[ix1]];
                        let y1_point = self.x.grid[1][[iy1]];
                        let x2_point = self.x.grid[2][[ind_border]];
                        let y2_point = self.x.grid[3][[iy2]];
                        let point = [x1_point, y1_point, x2_point, y2_point];
                        let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                        sum += j_flux[0] * normale[0]
                            + j_flux[1] * normale[1]
                            + j_flux[2] * normale[2]
                            + j_flux[3] * normale[3];
                    }
                }
            }
            sum * ds
        };

        let compute_y2_flow = |ind_border: usize, normale: [F; 4]| {
            let mut sum: C = C::new(0.0, 0.0);
            for ix1 in ind_min..ind_max {
                for iy1 in ind_min..ind_max {
                    for ix2 in ind_min..ind_max {
                        let x1_point = self.x.grid[0][[ix1]];
                        let y1_point = self.x.grid[1][[iy1]];
                        let x2_point = self.x.grid[2][[ix2]];
                        let y2_point = self.x.grid[3][[ind_border]];
                        let point = [x1_point, y1_point, x2_point, y2_point];
                        let j_flux = gauge.compute_flux(point, psi1, psi2, t);
                        sum += j_flux[0] * normale[0]
                            + j_flux[1] * normale[1]
                            + j_flux[2] * normale[2]
                            + j_flux[3] * normale[3];
                    }
                }
            }
            sum * ds
        };
        // Суммируем поток через все границы
        let x1_left_flow: C = compute_x1_flow(ind_min, [-1.0, 0.0, 0.0, 0.0]);
        let x1_right_flow: C = compute_x1_flow(ind_max, [1.0, 0.0, 0.0, 0.0]);
        let y1_left_flow: C = compute_y1_flow(ind_min, [0.0, -1.0, 0.0, 0.0]);
        let y1_right_flow: C = compute_y1_flow(ind_max, [0.0, 1.0, 0.0, 0.0]);
        let x2_left_flow: C = compute_x2_flow(ind_min, [0.0, 0.0, -1.0, 0.0]);
        let x2_right_flow: C = compute_x2_flow(ind_max, [0.0, 0.0, 1.0, 0.0]);
        let y2_left_flow: C = compute_y2_flow(ind_min, [0.0, 0.0, 0.0, -1.0]);
        let y2_right_flow: C = compute_y2_flow(ind_max, [0.0, 0.0, 0.0, 1.0]);

        x1_left_flow
            + x1_right_flow
            + y1_left_flow
            + y1_right_flow
            + x2_left_flow
            + x2_right_flow
            + y2_left_flow
            + y2_right_flow
    }
}

//============================================================================
//                 Плотность потока и калибровка
//============================================================================
/// Плотность потока вероятности в калибровке скорости
impl<'a> Flux<4> for VelocityGauge4D<'a> {
    fn compute_flux(
        &self,
        x: [F; 4],
        psi1: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
        psi2: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
        t: F,
    ) -> [C; 4] {
        let vec_pot = self.field.vec_pot(t); // векторный потенциал

        let psi1_val = psi1.value(x);
        let psi2_val = psi2.value(x);

        let psi1_derivs = psi1.deriv(x);
        let psi2_derivs = psi2.deriv(x);

        let zero_c = C::new(0.0, 0.0);
        let mut j: [C; 4] = [zero_c, zero_c, zero_c, zero_c];

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
pub struct Flow4D<'a, G: Flux<4> + Send + Sync + Copy, S: SurfaceFlow<4, G> + Sync> {
    gauge: &'a G,
    surface: &'a S,
    pub instance_flow: Vec<C>,
    pub time_instance: Vec<F>,
}

impl<'a, G: Flux<4> + Send + Sync + Copy, S: SurfaceFlow<4, G> + Sync> Flow4D<'a, G, S> {
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
        psi: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
        t: F,
    ) -> C {
        self.surface.compute_surface_flow(self.gauge, psi, psi, t)
        // .re // там только действительная часть должна остаться
    }

    pub fn add_instance_flow(
        &mut self,
        psi: &(impl ValueAndSpaceDerivatives<4> + Send + Sync),
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
