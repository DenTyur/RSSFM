use crate::field;
use crate::heatmap;
use crate::logcolormap;
use crate::parameters;
use field::Field2D;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::{Pspace, Xspace};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;

type F = f32;
type C = Complex<f32>;
const I: C = Complex::I;

pub trait ValueAndSpaceDerivatives {
    fn deriv(&self, x: [F; 2]) -> [C; 2];
    fn value(&self, x: [F; 2]) -> C;
}

pub struct WaveFunction<'a> {
    pub psi: Array<Complex<f32>, Ix2>,
    pub dpsi_dx: Array<Complex<f32>, Ix2>,
    pub dpsi_dy: Array<Complex<f32>, Ix2>,
    x: &'a Xspace,
    p: Pspace,
}

///спектральная производная
pub fn fft_dpsi_dx(psi: &mut Array2<C>, x: &Xspace, p: &Pspace) {
    use super::evolution::FftMaker2d;
    use itertools::multizip;
    let mut fft = FftMaker2d::new(&x.n);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // модифицируем psi
                    *psi_elem *= x.dx[0] * x.dx[1] / (2. * PI)
                        * (-I * (p.p0[0] * x_point + p.p0[1] * *y_point)).exp();
                });
        });
    fft.fft(psi);
    psi.axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut distr_row, px_point)| {
            distr_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(distr_elem, py_point)| {
                    *distr_elem *= I * px_point;
                });
        });
    fft.ifft(psi);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // демодифицируем psi
                    *psi_elem *= (2. * PI) / (x.dx[0] * x.dx[1])
                        * (I * (p.p0[0] * x_point + p.p0[1] * y_point)).exp();
                });
        });
}

pub fn fft_dpsi_dy(psi: &mut Array2<C>, x: &Xspace, p: &Pspace) {
    use super::evolution::FftMaker2d;
    use itertools::multizip;
    let mut fft = FftMaker2d::new(&x.n);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // модифицируем psi
                    *psi_elem *= x.dx[0] * x.dx[1] / (2. * PI)
                        * (-I * (p.p0[0] * x_point + p.p0[1] * *y_point)).exp();
                });
        });
    fft.fft(psi);
    psi.axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut distr_row, px_point)| {
            distr_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(distr_elem, py_point)| {
                    *distr_elem *= I * py_point;
                });
        });
    fft.ifft(psi);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // демодифицируем psi
                    *psi_elem *= (2. * PI) / (x.dx[0] * x.dx[1])
                        * (I * (p.p0[0] * x_point + p.p0[1] * y_point)).exp();
                });
        });
}

impl<'a> ValueAndSpaceDerivatives for WaveFunction<'a> {
    fn deriv(&self, x: [F; 2]) -> [C; 2] {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;

        // вычисляем производную
        [self.dpsi_dx[(ix, iy)], self.dpsi_dy[(ix, iy)]]
    }

    fn value(&self, x: [F; 2]) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;

        self.psi[(ix, iy)]
    }
}

impl<'a> WaveFunction<'a> {
    pub fn new(psi: Array<Complex<f32>, Ix2>, x: &'a Xspace) -> Self {
        let dpsi_dx: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let dpsi_dy: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let p = Pspace::init(x);
        Self {
            psi,
            x,
            p,
            dpsi_dx,
            dpsi_dy,
        }
    }

    pub fn update_derivatives(&mut self) {
        self.dpsi_dx = self.psi.clone();
        self.dpsi_dy = self.psi.clone();
        fft_dpsi_dx(&mut self.dpsi_dx, self.x, &self.p);
        fft_dpsi_dy(&mut self.dpsi_dy, self.x, &self.p);
    }

    // pub fn init_zeros(x: &'a Xspace) -> Self {
    //     let mut psi: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
    //     let j: Complex<f32> = Complex::I;
    //     Self { psi, x }
    // }
    //
    // // Инициализирует волновую функцию как двумерный осциллятор на основе пространственной сетки x
    // pub fn init_oscillator_2d(x: &'a Xspace) -> Self {
    //     let mut psi: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
    //     let j: Complex<f32> = Complex::I;
    //     psi.axis_iter_mut(Axis(0))
    //         .zip(x.grid[0].iter())
    //         .par_bridge()
    //         .for_each(|(mut psi_row, x_i)| {
    //             psi_row
    //                 .iter_mut()
    //                 .zip(x.grid[1].iter())
    //                 .for_each(|(psi_elem, y_j)| {
    //                     *psi_elem = (-0.5 * (x_i.powi(2) + y_j.powi(2))).exp() + 0. * j;
    //                 })
    //         });
    //     Self { psi, x }
    // }

    //Возвращает вероятность в расчетной области волновой функции
    pub fn prob_in_numerical_box(&self) -> f32 {
        let volume: f32 = self.x.dx[0] * self.x.dx[1];
        self.psi
            .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
            .sum_axis(Axis(0))
            .sum()
            * volume
    }

    //Возвращает норму волновой функции
    pub fn norm(&self) -> f32 {
        let volume: f32 = self.x.dx[0] * self.x.dx[1];
        // self.x.dx.iter().for_each(|dx| volume *= dx);
        f32::sqrt(
            self.psi
                .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
                .sum_axis(Axis(0))
                .sum()
                * volume,
        )
    }

    // Нормирует волновую функцию на 1
    pub fn normalization_by_1(&mut self) {
        let norm: f32 = self.norm();
        let j: Complex<f32> = Complex::I;
        self.psi *= (1. + 0. * j) / norm;
    }

    // Сохраняет волновую функцию в файл
    pub fn save_psi(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }

    // Загружает волновую функцию из файла.
    pub fn init_from_file(psi_path: &str, x: &'a Xspace) -> Self {
        let dpsi_dx: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let dpsi_dy: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let reader = File::open(psi_path).unwrap();
        let p = Pspace::init(x);
        Self {
            psi: Array::<Complex<f32>, Ix2>::read_npy(reader).unwrap(),
            x,
            p,
            dpsi_dx,
            dpsi_dy,
        }
    }

    pub fn lg_to_vgA(&mut self, field: &Field2D, t: f32) {
        let vec_pot = field.vec_pot(t);
        let b = field.b(t);

        self.psi
            .axis_iter_mut(Axis(0))
            .zip(self.x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, y)| {
                psi_row
                    .iter_mut()
                    .zip(self.x.grid[1].iter())
                    .for_each(|(psi_elem, x)| {
                        *psi_elem *= (-I * (x * vec_pot[0] + y * vec_pot[1]) + I / 2.0 * b).exp();
                    })
            });
    }

    pub fn vgA_to_lg(&mut self, field: &Field2D, t: f32) {
        let vec_pot = field.vec_pot(t);
        let b = field.b(t);

        self.psi
            .axis_iter_mut(Axis(0))
            .zip(self.x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, y)| {
                psi_row
                    .iter_mut()
                    .zip(self.x.grid[1].iter())
                    .for_each(|(psi_elem, x)| {
                        *psi_elem *= (I * (x * vec_pot[0] + y * vec_pot[1]) - I / 2.0 * b).exp();
                    })
            });
    }

    pub fn plot(&self, path: &str, colorbar_limits: [F; 2]) {
        let mut a: Array2<F> = Array::zeros((self.x.n[0], self.x.n[1]));

        self.psi
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
        let [colorbar_min, colorbar_max] = colorbar_limits;

        heatmap::plot_heatmap(
            &self.x.grid[0],
            &self.x.grid[1],
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
        let mut a: Array2<F> = Array::zeros((self.x.n[0], self.x.n[1]));

        self.psi
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
        let [colorbar_min, colorbar_max] = colorbar_limits;

        logcolormap::plot_heatmap_logscale(
            &a,
            &self.x.grid[0],
            &self.x.grid[1],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }
}
