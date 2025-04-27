use crate::field;
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
    x: &'a Xspace,
}

impl<'a> ValueAndSpaceDerivatives for WaveFunction<'a> {
    fn deriv(&self, x: [F; 2]) -> [C; 2] {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;

        // вычисляем радиальную производную
        let dpsi_dx = (self.psi[(ix + 1, iy)] - self.psi[(ix - 1, iy)]) / (2.0 * self.x.dx[0]);
        let dpsi_dy = (self.psi[(ix, iy + 1)] - self.psi[(ix, iy - 1)]) / (2.0 * self.x.dx[1]);

        [dpsi_dx, dpsi_dy]
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
        Self { psi, x }
    }

    pub fn init_zeros(x: &'a Xspace) -> Self {
        let mut psi: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let j: Complex<f32> = Complex::I;
        Self { psi, x }
    }

    // Инициализирует волновую функцию как двумерный осциллятор на основе пространственной сетки x
    pub fn init_oscillator_2d(x: &'a Xspace) -> Self {
        let mut psi: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let j: Complex<f32> = Complex::I;
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_i)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_j)| {
                        *psi_elem = (-0.5 * (x_i.powi(2) + y_j.powi(2))).exp() + 0. * j;
                    })
            });
        Self { psi, x }
    }

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
        let reader = File::open(psi_path).unwrap();
        Self {
            psi: Array::<Complex<f32>, Ix2>::read_npy(reader).unwrap(),
            x,
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
}
