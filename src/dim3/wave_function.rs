use super::space::{Pspace3D, Xspace3D};
use crate::config::{C, F, I};
use crate::dim2::{space::Xspace2D, wave_function::WaveFunction2D};
use crate::dim3::fft_maker::FftMaker3D;
use crate::macros::check_path;
use crate::traits::fft_maker::FftMaker;
use crate::traits::wave_function::{ValueAndSpaceDerivatives, WaveFunction};
use crate::utils::hdf5_interface;
use crate::utils::logcolormap;
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array3;
use ndarray::Ix3;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

/// Производные не реализованы, потому что для распределения ионов они не нужны.
/// Если надо будет считать импульсные распределения электронов или плотность потока вероятности, производные понадобятся.
/// Но вычисление производных это увеличение оперативной памяти в 5 раз (можно оптимизировать до 2
/// раз, но это надо заморочиться).
pub struct WaveFunction3D {
    pub psi: Array3<C>,
    pub dpsi_d0: Option<Array3<C>>,
    pub dpsi_d1: Option<Array3<C>>,
    pub dpsi_d2: Option<Array3<C>>,
    pub x: Xspace3D,
    pub p: Pspace3D,
}

impl WaveFunction3D {
    pub const DIM: usize = 3;

    pub fn new(psi: Array3<C>, x: &Xspace3D) -> Self {
        let p = Pspace3D::init(x);
        Self {
            psi,
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
            x: x.clone(),
            p,
        }
    }

    pub fn init_spectral_derivatives(&mut self) {
        self.dpsi_d0 = Some(self.psi.clone());
        self.dpsi_d1 = Some(self.psi.clone());
        self.dpsi_d2 = Some(self.psi.clone());
    }

    /// Инициализирует волновую функцию как трехмерный осциллятор на основе пространственной сетки x
    pub fn init_oscillator_3d(x: Xspace3D) -> Self {
        let mut psi: Array<C, Ix3> = Array::zeros((x.n[0], x.n[1], x.n[2]));
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_2d, x_0)| {
                psi_2d
                    .axis_iter_mut(Axis(0))
                    .zip(x.grid[1].iter())
                    .for_each(|(mut psi_1d, x_1)| {
                        psi_1d
                            .iter_mut()
                            .zip(x.grid[2].iter())
                            .for_each(|(psi_elem, x_2)| {
                                *psi_elem = (-0.5 * (x_0.powi(2) + x_1.powi(2) + x_2.powi(2)))
                                    .exp()
                                    + 0. * I;
                            })
                    })
            });
        let p = Pspace3D::init(&x);
        Self {
            psi,
            x,
            p,
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
        }
    }

    pub fn plot_slice_log(
        &self,
        path: &str,
        colorbar_limits: [F; 2],
        fixed_values: [Option<F>; 3], // None означает ось, по которой берется срез
    ) {
        // Преобразуем значения координат в индексы
        let fixed_indices: [Option<usize>; 3] = fixed_values.map(|val| {
            val.map(|v| {
                // Находим индекс для каждой оси
                self.x
                    .grid
                    .iter()
                    .enumerate()
                    .find_map(|(axis, grid)| {
                        if !grid.is_empty() {
                            let x_min = grid.first().unwrap();
                            let x_max = grid.last().unwrap();

                            // Проверяем, что значение в пределах сетки
                            if v < *x_min || v > *x_max {
                                panic!(
                                    "Value {} is out of bounds for axis {} (min: {}, max: {})",
                                    v, axis, x_min, x_max
                                );
                            }

                            // Аналитически вычисляем ближайший индекс
                            let idx = ((v - *x_min) / self.x.dx[axis]).round() as usize;

                            // Обеспечиваем, чтобы индекс был в допустимых пределах
                            Some(idx.min(grid.len() - 1))
                        } else {
                            None
                        }
                    })
                    .expect("Failed to find index")
            })
        });

        // Определяем, какие оси будут в срезе (те, для которых fixed_indices == None)
        let slice_axes: Vec<usize> = fixed_indices
            .iter()
            .enumerate()
            .filter(|(_, &idx)| idx.is_none())
            .map(|(i, _)| i)
            .collect();

        // Проверяем, что срез двумерный
        assert_eq!(slice_axes.len(), 2, "Slice must be 2D");

        // Создаем срез
        let slice = match fixed_indices {
            [None, None, Some(z)] => s![.., .., z],
            [None, Some(y), None] => s![.., y, ..],
            [Some(x), None, None] => s![x, .., ..],
            _ => panic!("Invalid slice configuration - exactly two axes must be None"),
        };

        // Применяем срез к пси-функции
        let sliced_psi = self.psi.slice(slice);

        // Создаем массив для плотности вероятности
        let shape: Vec<usize> = slice_axes.iter().map(|&axis| self.x.n[axis]).collect();
        let mut a: Array2<F> = Array::zeros((shape[0], shape[1]));
        sliced_psi
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
            &self.x.grid[slice_axes[0]],
            &self.x.grid[slice_axes[1]],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }
}

impl ValueAndSpaceDerivatives<3> for WaveFunction3D {
    fn deriv(&self, x: [F; Self::DIM]) -> [C; Self::DIM] {
        // unimplemented!("This method is not implemented yet");
        // нахождение индексов ближайших к окружности узлов сетки
        let x0_min = self.x.grid[0][[0]];
        let x1_min = self.x.grid[1][[0]];
        let x2_min = self.x.grid[2][[0]];
        let ix0 = ((x[0] - x0_min) / self.x.dx[0]).round() as usize;
        let ix1 = ((x[1] - x1_min) / self.x.dx[1]).round() as usize;
        let ix2 = ((x[2] - x2_min) / self.x.dx[2]).round() as usize;
        // возвращаем производную
        if self.dpsi_d0.is_none() || self.dpsi_d1.is_none() || self.dpsi_d2.is_none() {
            panic!("Derivatives are required but not available");
        }

        [
            self.dpsi_d0.as_ref().unwrap()[(ix0, ix1, ix2)],
            self.dpsi_d1.as_ref().unwrap()[(ix0, ix1, ix2)],
            self.dpsi_d2.as_ref().unwrap()[(ix0, ix1, ix2)],
        ]
    }

    fn value(&self, x: [F; Self::DIM]) -> C {
        // нахождение индексов ближайших к окружности узлов сетки
        let x_min = self.x.grid[0][[0]];
        let y_min = self.x.grid[1][[0]];
        let z_min = self.x.grid[2][[0]];
        let ix = ((x[0] - x_min) / self.x.dx[0]).round() as usize;
        let iy = ((x[1] - y_min) / self.x.dx[1]).round() as usize;
        let iz = ((x[2] - z_min) / self.x.dx[2]).round() as usize;
        // возвращаем значение
        self.psi[(ix, iy, iz)]
    }
}

impl WaveFunction<3> for WaveFunction3D {
    type Xspace = Xspace3D;

    /// Расширяет сетку волновой функции нулями
    fn extend(&mut self, x_new: &Xspace3D) {
        // Проверяем, что шаги сетки совпадают
        for i in 0..3 {
            assert!(
                (x_new.dx[i] - self.x.dx[i]).abs() < 1e-5,
                "Шаги сеток должны совпадать"
            );
        }

        // Определяем область пересечения старых и новых координат
        let mut x_start: [F; 3] = [0.0; 3];
        let mut x_end: [F; 3] = [0.0; 3];
        for i in 0..3 {
            x_start[i] = self.x.grid[i][0].max(x_new.grid[i][0]);
            x_end[i] = self.x.grid[i][self.x.n[i] - 1].min(x_new.grid[i][x_new.n[i] - 1]);
        }

        // Проверяем, что есть пересечение
        for i in 0..3 {
            assert!(
                x_start[i] <= x_end[i],
                "Нет пересечения между старой и новой сеткой"
            );
        }

        // Создаем новый массив, заполненный нулями
        let mut psi_new: Array3<C> = Array3::zeros((x_new.n[0], x_new.n[1], x_new.n[2]));

        // Находим индексы в новой сетке, куда нужно вставить данные
        let mut x_new_start_idx: [usize; 3] = [0; 3];
        let mut x_new_end_idx: [usize; 3] = [0; 3];
        for i in 0..3 {
            x_new_start_idx[i] = ((x_start[i] - x_new.grid[i][0]) / x_new.dx[i]).round() as usize;
            x_new_end_idx[i] = ((x_end[i] - x_new.grid[i][0]) / x_new.dx[i]).round() as usize + 1;
        }

        // Находим индексы в старой сетке, откуда брать данные
        let mut x_old_start_idx: [usize; 3] = [0; 3];
        let mut x_old_end_idx: [usize; 3] = [0; 3];
        for i in 0..3 {
            x_old_start_idx[i] = ((x_start[i] - self.x.grid[i][0]) / self.x.dx[i]).round() as usize;
            x_old_end_idx[i] = ((x_end[i] - self.x.grid[i][0]) / self.x.dx[i]).round() as usize + 1;
        }

        // Вставляем старые данные в новый массив
        // Копируем данные из старого массива в новый
        let mut new_slice = psi_new.slice_mut(s![
            x_new_start_idx[0]..x_new_end_idx[0],
            x_new_start_idx[1]..x_new_end_idx[1],
            x_new_start_idx[2]..x_new_end_idx[2],
        ]);
        let old_slice = self.psi.slice(s![
            x_old_start_idx[0]..x_old_end_idx[0],
            x_old_start_idx[1]..x_old_end_idx[1],
            x_old_start_idx[2]..x_old_end_idx[2],
        ]);
        new_slice.assign(&old_slice);

        // Обновляем все поля
        self.psi = psi_new;
        self.x = x_new.clone();
        self.p = Pspace3D::init(x_new);
    }

    fn update_derivatives(&mut self) {
        // unimplemented!("This method is not implemented yet");
        if self.dpsi_d0.is_none() || self.dpsi_d1.is_none() || self.dpsi_d2.is_none() {
            panic!("Derivatives are required but not available");
        }

        self.dpsi_d0 = Some(self.psi.clone());
        self.dpsi_d1 = Some(self.psi.clone());
        self.dpsi_d2 = Some(self.psi.clone());
        fft_dpsi_d0(self.dpsi_d0.as_mut().unwrap(), &self.x, &self.p);
        fft_dpsi_d1(self.dpsi_d1.as_mut().unwrap(), &self.x, &self.p);
        fft_dpsi_d2(self.dpsi_d2.as_mut().unwrap(), &self.x, &self.p);
    }

    fn prob_in_numerical_box(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1] * self.x.dx[2];
        self.psi.mapv(|a| (a.re.powi(2) + a.im.powi(2))).sum() * volume
    }

    fn norm(&self) -> F {
        let volume: F = self.x.dx[0] * self.x.dx[1] * self.x.dx[2];
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

    fn init_from_npy(psi_path: &str, x: Self::Xspace) -> Self {
        let reader = File::open(psi_path).unwrap();
        let p = Pspace3D::init(&x);
        Self {
            psi: Array::<C, Ix3>::read_npy(reader).unwrap(),
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
            x,
            p,
        }
    }

    fn init_from_hdf5(psi_path: &str) -> Self {
        let psi: Array3<C> =
            hdf5_interface::read_from_hdf5_complex(psi_path, "psi", Some("WaveFunction"))
                .unwrap()
                .mapv_into(|x| x as C); // преобразуем в нужный тип данных
        let x0: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x0", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let x1: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x1", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let x2: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x2", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let dx0 = x0[[1]] - x0[[0]];
        let dx1 = x1[[1]] - x1[[0]];
        let dx2 = x2[[1]] - x2[[0]];
        let xspace = Xspace3D {
            x0: [x0[[0]], x1[[0]], x2[[0]]],
            dx: [dx0, dx1, dx2],
            n: [x0.len(), x1.len(), x2.len()],
            grid: [x0, x1, x2],
        };

        let p = Pspace3D::init(&xspace);
        let dpsi_d0 = None;
        let dpsi_d1 = None;
        let dpsi_d2 = None;
        Self {
            psi,
            x: xspace,
            p,
            dpsi_d0,
            dpsi_d1,
            dpsi_d2,
        }
    }
}
//============================================================================================
/// спектральные производные

pub fn fft_dpsi_d0(psi: &mut Array3<C>, x: &Xspace3D, p: &Pspace3D) {
    let mut fft_maker = FftMaker3D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_2d, p0_point)| {
            multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_1d, _p1_point)| {
                    multizip((psi_1d.iter_mut(), p.grid[2].iter())).for_each(
                        |(elem, _p2_point)| {
                            *elem *= I * p0_point;
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}

pub fn fft_dpsi_d1(psi: &mut Array3<C>, x: &Xspace3D, p: &Pspace3D) {
    let mut fft_maker = FftMaker3D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_2d, _p0_point)| {
            multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_1d, p1_point)| {
                    multizip((psi_1d.iter_mut(), p.grid[2].iter())).for_each(
                        |(elem, _p2_point)| {
                            *elem *= I * p1_point;
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}

pub fn fft_dpsi_d2(psi: &mut Array3<C>, x: &Xspace3D, p: &Pspace3D) {
    let mut fft_maker = FftMaker3D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_2d, _p0_point)| {
            multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_1d, _p1_point)| {
                    multizip((psi_1d.iter_mut(), p.grid[2].iter())).for_each(|(elem, p2_point)| {
                        *elem *= I * p2_point;
                    });
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}
