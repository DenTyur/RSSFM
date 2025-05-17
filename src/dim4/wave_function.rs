use super::space::{Pspace4D, Xspace4D};
use crate::config::{C, F, I, PI};
use crate::macros::check_path;
use crate::traits::wave_function::{ValueAndSpaceDerivatives, WaveFunction};
use crate::utils::{heatmap, logcolormap};
use ndarray::prelude::*;
use ndarray::Array4;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

/// Производные не реализованы, потому что для распределения ионов они не нужны.
/// Если надо будет считать импульсные распределения электронов или плотность потока вероятности, производные понадобятся.
/// Но вычисление производных это увеличение оперативной памяти в 5 раз (можно оптимизировать до 2
/// раз, но это надо заморочиться).
pub struct WaveFunction4D {
    pub psi: Array4<C>,
    pub x: Xspace4D,
    pub p: Pspace4D,
}

impl WaveFunction4D {
    pub const DIM: usize = 4;

    pub fn new(psi: Array4<C>, x: &Xspace4D) -> Self {
        let p = Pspace4D::init(x);
        Self {
            psi,
            x: x.clone(),
            p,
        }
    }

    pub fn init_from_npy(psi_path: &str, x: Xspace4D) -> Self {
        let reader = File::open(psi_path).unwrap();
        let p = Pspace4D::init(&x);
        Self {
            psi: Array::<C, Ix4>::read_npy(reader).unwrap(),
            x,
            p,
        }
    }

    /// Расширяет сетку волновой функции нулями
    ///
    /// # Аргументы
    /// * `x_new` - структура Xspace4D c новыми координатными осями
    pub fn extend(&mut self, x_new: &Xspace4D) {
        for i in 0..4 {
            // Проверяем, что шаги сетки совпадают
            assert!(
                (x_new.dx[i] - self.x.dx[i]).abs() < 1e-10,
                "Шаг x и x_new должен совпадать"
            );
            // Проверяем, что новые оси содержат старые
            assert!(
                x_new.grid[i][0] <= self.x.grid[i][0]
                    && x_new.grid[i][x_new.n[i] - 1] >= self.x.grid[i][self.x.n[i] - 1],
                "x_new должна содержать x"
            );
        }

        // Создаем новый массив, заполненный нулями
        let mut psi_new: Array4<C> =
            Array4::zeros((x_new.n[0], x_new.n[1], x_new.n[2], x_new.n[3]));

        // Находим индексы, куда нужно вставить старый массив
        let x0_start = ((self.x.grid[0][0] - x_new.grid[0][0]) / x_new.dx[0]).round() as usize;
        let x0_end = x0_start + self.x.n[0];

        let x1_start = ((self.x.grid[1][0] - x_new.grid[1][0]) / x_new.dx[1]).round() as usize;
        let x1_end = x1_start + self.x.n[1];

        let x2_start = ((self.x.grid[2][0] - x_new.grid[2][0]) / x_new.dx[2]).round() as usize;
        let x2_end = x2_start + self.x.n[2];

        let x3_start = ((self.x.grid[3][0] - x_new.grid[3][0]) / x_new.dx[3]).round() as usize;
        let x3_end = x3_start + self.x.n[3];

        // Вставляем старые данные в новый массив
        let mut psi_slice = psi_new.slice_mut(s![
            x0_start..x0_end,
            x1_start..x1_end,
            x2_start..x2_end,
            x3_start..x3_end
        ]);
        psi_slice.assign(&self.psi);

        // Обновляем все поля
        self.psi = psi_new;
        self.x = x_new.clone();
        self.p = Pspace4D::init(x_new);
    }

    pub fn plot_slice_log(&self, path: &str, colorbar_limits: [F; 2]) {
        let mut a: Array2<F> = Array::zeros((self.x.n[0], self.x.n[1]));

        self.psi
            .slice(s![.., .., self.x.n[2] / 2 - 1, self.x.n[3] / 2 - 1])
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

impl ValueAndSpaceDerivatives<4> for WaveFunction4D {
    fn deriv(&self, x: [F; Self::DIM]) -> [C; Self::DIM] {
        unimplemented!("This method is not implemented yet");
    }

    fn value(&self, x: [F; Self::DIM]) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x1_min = self.x.grid[0][[0]];
        let y1_min = self.x.grid[1][[0]];
        let x2_min = self.x.grid[2][[0]];
        let y2_min = self.x.grid[3][[0]];
        let ix1 = ((x[0] - x1_min) / self.x.dx[0]).round() as usize;
        let iy1 = ((x[1] - y1_min) / self.x.dx[1]).round() as usize;
        let ix2 = ((x[2] - x2_min) / self.x.dx[2]).round() as usize;
        let iy2 = ((x[3] - y2_min) / self.x.dx[3]).round() as usize;
        // возвращаем значение
        self.psi[(ix1, iy1, ix2, iy2)]
    }
}

impl WaveFunction<4> for WaveFunction4D {
    fn update_derivatives(&mut self) {
        unimplemented!("This method is not implemented yet");
    }

    fn prob_in_numerical_box(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1] * self.x.dx[2] * self.x.dx[3];
        self.psi.mapv(|a| (a.re.powi(2) + a.im.powi(2))).sum() * volume
    }

    fn norm(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1] * self.x.dx[2] * self.x.dx[3];
        (self.psi.mapv(|a| (a.re.powi(2) + a.im.powi(2))).sum() * volume).sqrt()
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
