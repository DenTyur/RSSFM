use super::{
    fft_maker::FftMaker1D,
    space::{Pspace1D, Xspace1D},
};
use crate::config::{C, F, I, PI};
use crate::macros::check_path;
use crate::traits::{
    fft_maker::FftMaker,
    wave_function::{ValueAndSpaceDerivatives, WaveFunction},
};
use crate::utils::hdf5_interface;
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use plotters::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Clone)]
pub struct WaveFunction1D {
    pub psi: Array1<C>,
    pub dpsi_dx: Option<Array1<C>>,
    pub x: Xspace1D,
    pub p: Pspace1D,
}

impl WaveFunction1D {
    pub const DIM: usize = 1;

    pub fn new(psi: Array1<C>, x: Xspace1D) -> Self {
        let p = Pspace1D::init(&x);
        Self {
            psi,
            x,
            p,
            dpsi_dx: None,
        }
    }

    /// Инициализирует пустые массивы для спектральных производных
    pub fn init_spectral_derivatives(&mut self) {
        self.dpsi_dx = Some(self.psi.clone());
    }

    pub fn plot_log(&self, file_path: &str) {
        let x_values = self.x.grid[0].clone();
        let mut psi_norm_sq: Array1<F> = Array::zeros(self.x.n[0]);

        self.psi
            .iter()
            .zip(psi_norm_sq.iter_mut())
            .par_bridge()
            .for_each(|(psi_elem, a_elem)| {
                *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
            });
        // Создаем область для рисования
        let root = BitMapBackend::new(file_path, (1000, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Находим минимальное и максимальное значения для осей
        let x_min = x_values[[0]];
        let x_max = x_values[[self.x.n[0] - 1]];

        // Для логарифмической оси y находим диапазон значений (исключаем нули и отрицательные)
        let y_min = psi_norm_sq
            .iter()
            .filter(|&&y| y > 0.0)
            .fold(F::INFINITY, |a, &b| a.min(b))
            .log10();
        let y_max = psi_norm_sq
            .iter()
            .filter(|&&y| y > 0.0)
            .fold(F::NEG_INFINITY, |a, &b| a.max(b))
            .log10();

        // Создаем график с линейной осью x и логарифмической осью y
        let mut chart = ChartBuilder::on(&root)
            .caption("|psi|^2 (log scale)", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(40)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        // Настраиваем ось y как логарифмическую
        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("log10(|psi|^2)")
            .y_label_formatter(&|y| format!("10^{:.1}", y))
            .draw()
            .unwrap();

        // Рисуем график
        chart
            .draw_series(LineSeries::new(
                x_values
                    .iter()
                    .zip(psi_norm_sq.iter())
                    .map(|(x, y)| (*x, y.log10())),
                &BLUE,
            ))
            .unwrap();
    }

    pub fn plot(&self, path: &str, limits: [F; 2]) {
        let mut a: Array1<F> = Array::zeros(self.x.n[0]);

        self.psi
            .iter()
            .zip(a.iter_mut())
            .par_bridge()
            .for_each(|(psi_elem, a_elem)| {
                *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
            });

        check_path!(path);

        // Создаём область для графика
        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Автоматически вычисляем диапазоны осей
        let x_range = self.x.grid[0]
            .iter()
            .fold(F::INFINITY..F::NEG_INFINITY, |range, &val| {
                val.min(range.start)..val.max(range.end)
            });

        // let y_range = self
        //     .instance_flow
        //     .iter()
        //     .fold(F::INFINITY..F::NEG_INFINITY, |range, &val| {
        //         val.re.min(range.start)..val.re.max(range.end)
        //     });

        // Добавляем 10% отличия по краям для лучшего отображения
        // let x_padding = (x_range.end - x_range.start) * 0.1;
        let y_padding = (limits[0] - limits[1]) * 0.1;

        let x_range = x_range.start..x_range.end;
        let y_range = (limits[0] - y_padding)..(limits[1] + y_padding);

        // Создаём график
        let mut chart = ChartBuilder::on(&root)
            .caption("|psi|^2", ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(x_range, y_range)
            .unwrap();

        // Настраиваем сетку
        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("|psi|")
            .draw()
            .unwrap();

        // Abs sq Psi
        // Рисуем линию графика
        chart
            .draw_series(LineSeries::new(
                self.x.grid[0]
                    .iter()
                    .zip(self.psi.iter())
                    .map(|(&x, &y)| (x, y.norm())),
                BLUE.stroke_width(1),
            ))
            .unwrap();
    }
}

impl ValueAndSpaceDerivatives<1> for WaveFunction1D {
    fn deriv(&self, x: [F; Self::DIM], axis: usize) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        // возвращаем производную
        if self.dpsi_dx.is_none() {
            panic!("Derivatives are required but not available");
        }

        match axis {
            0 => self
                .dpsi_dx
                .as_ref()
                .expect("Derivatives are required but not available")[ix],
            _ => unreachable!("axis > DIM"),
        }
    }

    fn value(&self, x: [F; Self::DIM]) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        // возвращаем значение
        self.psi[ix]
    }
}

impl WaveFunction<1> for WaveFunction1D {
    type Xspace = Xspace1D;

    fn extend(&mut self, x_new: &Xspace1D) {
        // Проверяем, что шаги сетки совпадают
        assert!(
            (x_new.dx[0] - self.x.dx[0]).abs() < 1e-5,
            "Шаг x должен совпадать"
        );

        // Определяем область пересечения старых и новых координат
        let x_start = self.x.grid[0][0].max(x_new.grid[0][0]);
        let x_end = self.x.grid[0][self.x.n[0] - 1].min(x_new.grid[0][x_new.n[0] - 1]);

        // Проверяем, что есть пересечение
        assert!(
            x_start <= x_end,
            "Нет пересечения между старой и новой сеткой"
        );

        // Создаем новый массив, заполненный нулями
        let mut psi_new: Array1<C> = Array1::zeros(x_new.n[0]);

        // Находим индексы в новой сетке, куда нужно вставить данные
        let new_start_idx = ((x_start - x_new.grid[0][0]) / x_new.dx[0]).round() as usize;
        let new_end_idx = ((x_end - x_new.grid[0][0]) / x_new.dx[0]).round() as usize + 1;

        // Находим индексы в старой сетке, откуда брать данные
        let old_start_idx = ((x_start - self.x.grid[0][0]) / self.x.dx[0]).round() as usize;
        let old_end_idx = ((x_end - self.x.grid[0][0]) / self.x.dx[0]).round() as usize + 1;

        // Копируем данные из старого массива в новый
        let mut new_slice = psi_new.slice_mut(s![new_start_idx..new_end_idx]);
        let old_slice = self.psi.slice(s![old_start_idx..old_end_idx]);
        new_slice.assign(&old_slice);

        // Обновляем все поля
        self.psi = psi_new;
        self.dpsi_dx = Some(Array1::zeros(x_new.n[0]));
        self.x = x_new.clone();
        self.p = Pspace1D::init(x_new);
    }

    fn update_derivatives(&mut self) {
        self.dpsi_dx = Some(self.psi.clone());
        if let Some(ref mut dpsi) = self.dpsi_dx {
            // Разворачиваем Option
            fft_dpsi_dx(dpsi, &self.x, &self.p);
        }
    }

    fn prob_in_numerical_box(&self) -> F {
        let volume: F = self.x.dx[0];
        self.psi
            .mapv(|a| (a.re.powi(2) + a.im.powi(2)))
            .sum_axis(Axis(0))
            .sum()
            * volume
    }

    fn norm(&self) -> F {
        let volume: F = self.x.dx[0];
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
        let reader = File::open(psi_path).unwrap();
        let p = Pspace1D::init(&x);
        Self {
            psi: Array::<C, Ix1>::read_npy(reader).unwrap(),
            x,
            p,
            dpsi_dx: None,
        }
    }

    fn init_from_hdf5(psi_path: &str) -> Self {
        let psi: Array1<C> =
            hdf5_interface::read_from_hdf5_complex(psi_path, "psi", Some("WaveFunction"))
                .unwrap()
                .mapv_into(|x| x as C); // преобразуем в нужный тип данных
        let x0: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x0", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let dx = x0[[1]] - x0[[0]];
        let xspace = Xspace1D {
            x0: [x0[[0]]],
            dx: [dx],
            n: [x0.len()],
            grid: [x0],
        };

        let p = Pspace1D::init(&xspace);
        Self {
            psi,
            x: xspace,
            p,
            dpsi_dx: None,
        }
    }
}
//=================================================================================
///спектральная производная -- всратость неимоверная, оптимизировать!
pub fn fft_dpsi_dx(psi: &mut Array1<C>, x: &Xspace1D, p: &Pspace1D) {
    let mut fft_maker = FftMaker1D::new(&x.n);
    multizip((psi.iter_mut(), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(psi_elem, x_point)| {
            // модифицируем psi
            *psi_elem *= x.dx[0] / (2. * PI).sqrt() * (-I * (p.p0[0] * x_point)).exp();
        });
    fft_maker.fft(psi);
    psi.iter_mut()
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(distr_elem, px_point)| {
            *distr_elem *= I * px_point;
        });
    fft_maker.ifft(psi);
    multizip((psi.iter_mut(), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(psi_elem, x_point)| {
            *psi_elem *= (2. * PI).sqrt() / x.dx[0] * (I * (p.p0[0] * x_point)).exp();
        });
}
