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
use crate::utils::hdf5_interface;
use crate::utils::{heatmap, logcolormap};
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Clone)]
pub struct WaveFunction2D {
    pub psi: Array2<C>,
    pub dpsi_dx: Option<Array2<C>>,
    pub dpsi_dy: Option<Array2<C>>,
    pub x: Xspace2D,
    pub p: Pspace2D,
}

impl WaveFunction2D {
    pub const DIM: usize = 2;

    pub fn new(psi: Array2<C>, x: Xspace2D) -> Self {
        let p = Pspace2D::init(&x);
        Self {
            psi,
            x,
            p,
            dpsi_dx: None,
            dpsi_dy: None,
        }
    }

    /// Инициализирует пустые массивы для спектральных производных
    pub fn init_spectral_derivatives(&mut self) {
        self.dpsi_dx = Some(self.psi.clone());
        self.dpsi_dy = Some(self.psi.clone());
    }

    /// Инициализирует волновую функцию как двумерный осциллятор на основе пространственной сетки x
    pub fn init_oscillator_2d(x: Xspace2D) -> Self {
        let mut psi: Array<C, Ix2> = Array::zeros((x.n[0], x.n[1]));
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_i)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_j)| {
                        *psi_elem = (-0.5 * (x_i.powi(2) + y_j.powi(2))).exp() + 0. * I;
                    })
            });
        let p = Pspace2D::init(&x);
        Self {
            psi,
            x,
            p,
            dpsi_dx: None,
            dpsi_dy: None,
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
        if self.dpsi_dx.is_none() || self.dpsi_dy.is_none() {
            panic!("Derivatives are required but not available");
        }

        [
            self.dpsi_dx.as_ref().unwrap()[(ix, iy)],
            self.dpsi_dy.as_ref().unwrap()[(ix, iy)],
        ]
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
    type Xspace = Xspace2D;

    fn extend(&mut self, x_new: &Xspace2D) {
        // Проверяем, что шаги сетки совпадают
        assert!(
            (x_new.dx[0] - self.x.dx[0]).abs() < 1e-5,
            "Шаг x0 должен совпадать"
        );
        assert!(
            (x_new.dx[1] - self.x.dx[1]).abs() < 1e-5,
            "Шаг x1 должен совпадать"
        );

        // Определяем область пересечения старых и новых координат
        let x_start = self.x.grid[0][0].max(x_new.grid[0][0]);
        let x_end = self.x.grid[0][self.x.n[0] - 1].min(x_new.grid[0][x_new.n[0] - 1]);

        let y_start = self.x.grid[1][0].max(x_new.grid[1][0]);
        let y_end = self.x.grid[1][self.x.n[1] - 1].min(x_new.grid[1][x_new.n[1] - 1]);

        // Проверяем, что есть пересечение
        assert!(
            x_start <= x_end,
            "Нет пересечения между старой и новой сеткой x"
        );
        assert!(
            y_start <= y_end,
            "Нет пересечения между старой и новой сеткой y"
        );

        // Создаем новый массив, заполненный нулями
        let mut psi_new: Array2<C> = Array2::zeros((x_new.n[0], x_new.n[1]));

        // Находим индексы в новой сетке, куда нужно вставить данные
        let x_new_start_idx = ((x_start - x_new.grid[0][0]) / x_new.dx[0]).round() as usize;
        let x_new_end_idx = ((x_end - x_new.grid[0][0]) / x_new.dx[0]).round() as usize + 1;

        let y_new_start_idx = ((y_start - x_new.grid[1][0]) / x_new.dx[1]).round() as usize;
        let y_new_end_idx = ((y_end - x_new.grid[1][0]) / x_new.dx[1]).round() as usize + 1;

        // Находим индексы в старой сетке, откуда брать данные
        let x_old_start_idx = ((x_start - self.x.grid[0][0]) / self.x.dx[0]).round() as usize;
        let x_old_end_idx = ((x_end - self.x.grid[0][0]) / self.x.dx[0]).round() as usize + 1;

        let y_old_start_idx = ((y_start - self.x.grid[1][0]) / self.x.dx[1]).round() as usize;
        let y_old_end_idx = ((y_end - self.x.grid[1][0]) / self.x.dx[1]).round() as usize + 1;

        // Вставляем старые данные в новый массив
        // Копируем данные из старого массива в новый
        let mut new_slice = psi_new.slice_mut(s![
            x_new_start_idx..x_new_end_idx,
            y_new_start_idx..y_new_end_idx
        ]);
        let old_slice = self.psi.slice(s![
            x_old_start_idx..x_old_end_idx,
            y_old_start_idx..y_old_end_idx
        ]);
        new_slice.assign(&old_slice);

        // Обновляем все поля
        self.psi = psi_new;
        self.dpsi_dx = Array2::zeros((x_new.n[0], x_new.n[1]));
        self.dpsi_dy = Array2::zeros((x_new.n[0], x_new.n[1]));
        self.x = x_new.clone();
        self.p = Pspace2D::init(x_new);
    }

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
        check_path!(path);
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }

    fn init_from_npy(psi_path: &str, x: Self::Xspace) -> Self {
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

    fn init_from_hdf5(psi_path: &str) -> Self {
        let psi: Array2<C> =
            hdf5_interface::read_from_hdf5_complex(psi_path, "psi", Some("WaveFunction"))
                .unwrap()
                .mapv_into(|x| x as C); // преобразуем в нужный тип данных
        let x0: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x0", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let x1: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x1", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let dx0 = x0[[1]] - x0[[0]];
        let dx1 = x1[[1]] - x1[[0]];
        let xspace = Xspace2D {
            x0: [x0[[0]], x1[[0]]],
            dx: [dx0, dx1],
            n: [x0.len(), x1.len()],
            grid: [x0, x1],
        };

        let p = Pspace2D::init(&xspace);
        let dpsi_dx: Array2<C> = Array::zeros((xspace.n[0], xspace.n[1]));
        let dpsi_dy: Array2<C> = Array::zeros((xspace.n[0], xspace.n[1]));
        Self {
            psi,
            x: xspace,
            p,
            dpsi_dx,
            dpsi_dy,
        }
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
