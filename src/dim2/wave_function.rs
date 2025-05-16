use super::{
    fft_maker::FftMaker2D,
    space::{Pspace2D, Xspace2D},
};
use crate::config::{C, F, I, PI};
use crate::macros::check_path;
use crate::traits::{
    fft_maker::FftMaker,
    wave_function::{ValueAndSpaceDerivatives, WaveFunction},
};
use crate::utils::{heatmap, logcolormap};
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

pub struct WaveFunction2D {
    pub psi: Array2<C>,
    pub dpsi_dx: Array2<C>,
    pub dpsi_dy: Array2<C>,
    pub x: Xspace2D,
    pub p: Pspace2D,
}

impl WaveFunction2D {
    pub const DIM: usize = 2;

    pub fn new(psi: Array2<C>, x: &Xspace2D) -> Self {
        let dpsi_dx: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        let dpsi_dy: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        let p = Pspace2D::init(x);
        Self {
            psi,
            x: x.clone(),
            p,
            dpsi_dx,
            dpsi_dy,
        }
    }

    pub fn init_from_npy(psi_path: &str, x: Xspace2D) -> Self {
        let dpsi_dx: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        let dpsi_dy: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        let reader = File::open(psi_path).unwrap();
        let p = Pspace2D::init(&x);
        Self {
            psi: Array::<C, Ix2>::read_npy(reader).unwrap(),
            x,
            p,
            dpsi_dx,
            dpsi_dy,
        }
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

        check_path!(path);
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

        let [colorbar_min, colorbar_max] = colorbar_limits;

        check_path!(path);
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

impl ValueAndSpaceDerivatives<2> for WaveFunction2D {
    fn deriv(&self, x: [F; Self::DIM]) -> [C; Self::DIM] {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;
        // возвращаем производную
        [self.dpsi_dx[(ix, iy)], self.dpsi_dy[(ix, iy)]]
    }

    fn value(&self, x: [F; Self::DIM]) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;
        // возвращаем значение
        self.psi[(ix, iy)]
    }
}

impl WaveFunction<2> for WaveFunction2D {
    fn update_derivatives(&mut self) {
        self.dpsi_dx = self.psi.clone();
        self.dpsi_dy = self.psi.clone();
        fft_dpsi_dx(&mut self.dpsi_dx, &self.x, &self.p);
        fft_dpsi_dy(&mut self.dpsi_dy, &self.x, &self.p);
    }

    fn prob_in_numerical_box(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1];
        self.psi
            .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
            .sum_axis(Axis(0))
            .sum()
            * volume
    }

    fn norm(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1];
        (self
            .psi
            .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
            .sum_axis(Axis(0))
            .sum()
            * volume)
            .sqrt()
    }

    fn normalization_by_1(&mut self) {
        let norm: F = self.norm();
        let j: C = Complex::I;
        self.psi *= (1. + 0. * j) / norm;
    }

    fn save_as_npy(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }
}
//=================================================================================
///спектральная производная -- всратость неимоверная, оптимизировать!
pub fn fft_dpsi_dx(psi: &mut Array2<C>, x: &Xspace2D, p: &Pspace2D) {
    let mut fft_maker = FftMaker2D::new(&x.n);
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
    fft_maker.fft(psi);
    psi.axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut distr_row, px_point)| {
            distr_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(distr_elem, _py_point)| {
                    *distr_elem *= I * px_point;
                });
        });
    fft_maker.ifft(psi);
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

pub fn fft_dpsi_dy(psi: &mut Array2<C>, x: &Xspace2D, p: &Pspace2D) {
    let mut fft_maker = FftMaker2D::new(&x.n);
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
    fft_maker.fft(psi);
    psi.axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut distr_row, _px_point)| {
            distr_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(distr_elem, py_point)| {
                    *distr_elem *= I * py_point;
                });
        });
    fft_maker.ifft(psi);
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
