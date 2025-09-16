use super::space::{Pspace4D, Xspace4D};
use crate::config::{C, F, I};
use crate::dim2::{space::Xspace2D, wave_function::WaveFunction2D};
use crate::dim4::fft_maker::FftMaker4D;
use crate::macros::check_path;
use crate::traits::fft_maker::FftMaker;
use crate::traits::wave_function::{ValueAndSpaceDerivatives, WaveFunction};
use crate::utils::hdf5_interface;
use crate::utils::logcolormap;
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array4;
use ndarray::Ix4;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::{BufRead, BufReader, Error, Write};

/// Перечисления для указания, вкаком представлении находитсяволновая функция
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    Position, // координатное представление
    Momentum, // импульсное представление
}

/// Производные не реализованы, потому что для распределения ионов они не нужны.
/// Если надо будет считать импульсные распределения электронов или плотность потока вероятности, производные понадобятся.
/// Но вычисление производных это увеличение оперативной памяти в 5 раз (можно оптимизировать до 2
/// раз, но это надо заморочиться).
pub struct WaveFunction4D {
    pub psi: Array4<C>,
    pub dpsi_d0: Option<Array4<C>>,
    pub dpsi_d1: Option<Array4<C>>,
    pub dpsi_d2: Option<Array4<C>>,
    pub dpsi_d3: Option<Array4<C>>,
    pub x: Xspace4D,
    pub p: Pspace4D,
    pub representation: Representation,
}

impl WaveFunction4D {
    pub const DIM: usize = 4;

    pub fn new(psi: Array4<C>, x: &Xspace4D) -> Self {
        let p = Pspace4D::init(x);
        Self {
            psi,
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
            dpsi_d3: None,
            x: x.clone(),
            p,
            representation: Representation::Position,
        }
    }

    pub fn init_spectral_derivatives(&mut self) {
        self.dpsi_d0 = Some(self.psi.clone());
        self.dpsi_d1 = Some(self.psi.clone());
        self.dpsi_d2 = Some(self.psi.clone());
        self.dpsi_d3 = Some(self.psi.clone());
    }
}

// Работа с центром масс
impl WaveFunction4D {
    /// Возвращает волновую функцию центра масс
    /// Для взаимодействующих электронов это не совсем правильно,
    /// так как волновая функци не факторизуется на относительную в.ф.
    /// и в.ф. центра масс.
    pub fn get_psi_center_of_mass(&self) -> WaveFunction2D {
        use std::collections::HashMap;
        let d2x = self.x.dx[2] * self.x.dx[3];

        // Сетка для центра масс (X, Y) совпадает с исходной
        let X_grid = self.x.grid[0].clone();
        let Y_grid = self.x.grid[1].clone();

        // Массив для psi_cm(X, Y)
        let mut psi_cm: Array2<C> = Array2::zeros((self.x.n[0], self.x.n[1]));

        let tolerance: F = 1e-6;
        // Предварительно строим HashMap для быстрого поиска индексов
        let index_map: HashMap<_, _> = self.x.grid[0]
            .iter()
            .enumerate()
            .map(|(i, &val)| (val.to_bits(), i))
            .collect();

        let get_index = |x: F| -> Option<usize> { index_map.get(&x.to_bits()).copied() };

        psi_cm
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let X = X_grid[i];

                for (j, psi_cm_ij) in row.iter_mut().enumerate() {
                    let Y = Y_grid[j];
                    let mut sum: C = C::new(0.0, 0.0);

                    // Оптимизированный перебор по r_x и r_y
                    for &delta_x in self.x.grid[0].iter() {
                        let r_x = delta_x - self.x.grid[0][0];
                        for sign_x in [-1.0, 1.0].iter() {
                            let x1 = X + sign_x * r_x / 2.0;
                            let x2 = X - sign_x * r_x / 2.0;

                            let Some(x1_idx) = get_index(x1) else {
                                continue;
                            };
                            let Some(x2_idx) = get_index(x2) else {
                                continue;
                            };

                            for &delta_y in self.x.grid[1].iter() {
                                let r_y = delta_y - self.x.grid[1][0];
                                for sign_y in [-1.0, 1.0].iter() {
                                    let y1 = Y + sign_y * r_y / 2.0;
                                    let y2 = Y - sign_y * r_y / 2.0;

                                    let Some(y1_idx) = get_index(y1) else {
                                        continue;
                                    };
                                    let Some(y2_idx) = get_index(y2) else {
                                        continue;
                                    };

                                    sum += self.psi[[x1_idx, y1_idx, x2_idx, y2_idx]];
                                }
                            }
                        }
                    }

                    *psi_cm_ij = sum * self.x.dx[2] * self.x.dx[3] / 4.0;
                }
            });

        let X_cm = Xspace2D {
            x0: [X_grid[0], Y_grid[0]],
            dx: [self.x.dx[0], self.x.dx[1]],
            n: [X_grid.len(), Y_grid.len()],
            grid: [X_grid, Y_grid],
        };
        WaveFunction2D::new(psi_cm, X_cm)
    }
}

/// Графическая обработка
impl WaveFunction4D {
    pub fn plot_slice_log(
        &self,
        path: &str,
        colorbar_limits: [F; 2],
        fixed_values: [Option<F>; 4], // None означает ось, по которой берется срез
    ) {
        // Определяем, импульсное представление или координатное
        let (grid, grid_first_elem, d_grid, grid_n) = match self.representation {
            Representation::Position => (&self.x.grid, &self.x.x0, &self.x.dx, &self.x.n),
            Representation::Momentum => (&self.p.grid, &self.p.p0, &self.p.dp, &self.x.n),
        };

        // Теперь используем переменные дальше в коде
        let mut fixed_indices: [Option<usize>; 4] = [None; 4];
        for i in 0..4 {
            if fixed_values[i] != None {
                let ind =
                    ((fixed_values[i].unwrap() - grid_first_elem[i]) / d_grid[i]).round() as usize;
                fixed_indices[i] = Some(ind);
            }
        }

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
            [None, None, Some(z), Some(w)] => s![.., .., z, w],
            [None, Some(y), None, Some(w)] => s![.., y, .., w],
            [None, Some(y), Some(z), None] => s![.., y, z, ..],
            [Some(x), None, None, Some(w)] => s![x, .., .., w],
            [Some(x), None, Some(z), None] => s![x, .., z, ..],
            [Some(x), Some(y), None, None] => s![x, y, .., ..],
            _ => panic!("Invalid slice configuration - exactly two axes must be None"),
        };

        // Применяем срез к пси-функции
        let sliced_psi = self.psi.slice(slice);

        // Создаем массив для плотности вероятности
        let shape: Vec<usize> = slice_axes.iter().map(|&axis| grid_n[axis]).collect();
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
                        *a_elem = psi_elem.norm_sqr(); //.im.powi(2) + psi_elem.re.powi(2);
                    })
            });

        let [colorbar_min, colorbar_max] = colorbar_limits;

        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &a,
            &grid[slice_axes[0]],
            &grid[slice_axes[1]],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }
}

impl ValueAndSpaceDerivatives<4> for WaveFunction4D {
    fn deriv(&self, x: [F; Self::DIM], axis: usize) -> C {
        // unimplemented!("This method is not implemented yet");
        // нахождение индексов ближайших к окружности узлов сетки
        let x0_min = self.x.grid[0][[0]];
        let x1_min = self.x.grid[1][[0]];
        let x2_min = self.x.grid[2][[0]];
        let x3_min = self.x.grid[3][[0]];
        let ix0 = ((x[0] - x0_min) / self.x.dx[0]).round() as usize;
        let ix1 = ((x[1] - x1_min) / self.x.dx[1]).round() as usize;
        let ix2 = ((x[2] - x2_min) / self.x.dx[2]).round() as usize;
        let ix3 = ((x[3] - x3_min) / self.x.dx[3]).round() as usize;
        // возвращаем производную
        if self.dpsi_d0.is_none()
            || self.dpsi_d1.is_none()
            || self.dpsi_d2.is_none()
            || self.dpsi_d3.is_none()
        {
            panic!("Derivatives are required but not available");
        }

        match axis {
            0 => self
                .dpsi_d0
                .as_ref()
                .expect("Derivatives are required but not available")[(ix0, ix1, ix2, ix3)],
            1 => self
                .dpsi_d1
                .as_ref()
                .expect("Derivatives are required but not available")[(ix0, ix1, ix2, ix3)],
            2 => self
                .dpsi_d2
                .as_ref()
                .expect("Derivatives are required but not available")[(ix0, ix1, ix2, ix3)],
            3 => self
                .dpsi_d3
                .as_ref()
                .expect("Derivatives are required but not available")[(ix0, ix1, ix2, ix3)],
            _ => panic!("deriv 4d: axis>3 !"),
        }
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
    type Xspace = Xspace4D;

    /// Расширяет сетку волновой функции нулями
    fn extend(&mut self, x_new: &Xspace4D) {
        // Проверяем, что шаги сетки совпадают
        for i in 0..4 {
            assert!(
                (x_new.dx[i] - self.x.dx[i]).abs() < 1e-5,
                "Шаги сеток должены совпадать"
            );
        }

        // Определяем область пересечения старых и новых координат
        let mut x_start: [F; 4] = [0.0; 4];
        let mut x_end: [F; 4] = [0.0; 4];
        for i in 0..4 {
            x_start[i] = self.x.grid[i][0].max(x_new.grid[i][0]);
            x_end[i] = self.x.grid[i][self.x.n[i] - 1].min(x_new.grid[i][x_new.n[i] - 1]);
        }

        // Проверяем, что есть пересечение
        for i in 0..4 {
            assert!(
                x_start[i] <= x_end[i],
                "Нет пересечения между старой и новой сеткой"
            );
        }

        // Создаем новый массив, заполненный нулями
        let mut psi_new: Array4<C> =
            Array4::zeros((x_new.n[0], x_new.n[1], x_new.n[2], x_new.n[3]));

        // Находим индексы в новой сетке, куда нужно вставить данные
        let mut x_new_start_idx: [usize; 4] = [0; 4];
        let mut x_new_end_idx: [usize; 4] = [0; 4];
        for i in 0..4 {
            x_new_start_idx[i] = ((x_start[i] - x_new.grid[i][0]) / x_new.dx[i]).round() as usize;
            x_new_end_idx[i] = ((x_end[i] - x_new.grid[i][0]) / x_new.dx[i]).round() as usize + 1;
        }

        // Находим индексы в старой сетке, откуда брать данные
        let mut x_old_start_idx: [usize; 4] = [0; 4];
        let mut x_old_end_idx: [usize; 4] = [0; 4];
        for i in 0..4 {
            x_old_start_idx[i] = ((x_start[i] - self.x.grid[i][0]) / self.x.dx[i]).round() as usize;
            x_old_end_idx[i] = ((x_end[i] - self.x.grid[i][0]) / self.x.dx[i]).round() as usize + 1;
        }

        // Вставляем старые данные в новый массив
        // Копируем данные из старого массива в новый
        let mut new_slice = psi_new.slice_mut(s![
            x_new_start_idx[0]..x_new_end_idx[0],
            x_new_start_idx[1]..x_new_end_idx[1],
            x_new_start_idx[2]..x_new_end_idx[2],
            x_new_start_idx[3]..x_new_end_idx[3],
        ]);
        let old_slice = self.psi.slice(s![
            x_old_start_idx[0]..x_old_end_idx[0],
            x_old_start_idx[1]..x_old_end_idx[1],
            x_old_start_idx[2]..x_old_end_idx[2],
            x_old_start_idx[3]..x_old_end_idx[3],
        ]);
        new_slice.assign(&old_slice);

        // Обновляем все поля
        self.psi = psi_new;
        self.x = x_new.clone();
        self.p = Pspace4D::init(x_new);
    }

    fn update_derivatives(&mut self) {
        if self.dpsi_d0.is_none()
            || self.dpsi_d1.is_none()
            || self.dpsi_d2.is_none()
            || self.dpsi_d3.is_none()
        {
            panic!("Derivatives are required but not available");
        }

        self.dpsi_d0 = Some(self.psi.clone());
        self.dpsi_d1 = Some(self.psi.clone());
        self.dpsi_d2 = Some(self.psi.clone());
        self.dpsi_d3 = Some(self.psi.clone());
        fft_dpsi_d0(self.dpsi_d0.as_mut().unwrap(), &self.x, &self.p);
        fft_dpsi_d1(self.dpsi_d1.as_mut().unwrap(), &self.x, &self.p);
        fft_dpsi_d2(self.dpsi_d2.as_mut().unwrap(), &self.x, &self.p);
        fft_dpsi_d3(self.dpsi_d3.as_mut().unwrap(), &self.x, &self.p);
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
        check_path!(path);
        let writer = BufWriter::new(File::create(path)?);
        self.psi.write_npy(writer)?;
        Ok(())
    }

    /// Saves a sparsed slice of the 4D wave function to an NPY file and stores the sparse step.
    fn save_sparsed_as_npy(&self, path: &str, sparse_step: isize) -> Result<(), WriteNpyError> {
        check_path!(path);

        // Extract directory from path, default to current directory if none
        let dir_path = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::Path::new(".").to_path_buf());

        // Create path for slice step metadata file
        let step_file_path = dir_path.join("sparse_step.txt");

        // Save slice step value to metadata file
        let mut output = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&step_file_path)
            .map_err(|e| {
                WriteNpyError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to open {}: {}", step_file_path.display(), e),
                ))
            })?;

        write!(output, "{}", sparse_step).map_err(|e| {
            WriteNpyError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to write to {}: {}", step_file_path.display(), e),
            ))
        })?;

        // Save sparsed wave function slice to NPY file
        let writer = BufWriter::new(File::create(path).map_err(|e| {
            WriteNpyError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create {}: {}", path, e),
            ))
        })?);

        self.psi
            .slice(s![..;sparse_step,..;sparse_step,..;sparse_step,..;sparse_step])
            .write_npy(writer)?;

        Ok(())
    }

    fn init_from_npy(psi_path: &str, x: Self::Xspace) -> Self {
        let reader = File::open(psi_path).unwrap();
        let p = Pspace4D::init(&x);
        Self {
            psi: Array::<C, Ix4>::read_npy(reader).unwrap(),
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
            dpsi_d3: None,
            x,
            p,
            representation: Representation::Position,
        }
    }

    fn init_from_hdf5(psi_path: &str) -> Self {
        let psi: Array4<C> =
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
        let x3: Array1<F> = hdf5_interface::read_from_hdf5(psi_path, "x3", Some("Xspace"))
            .unwrap()
            .mapv_into(|x| x as F); // преобразуем в нужный тип данных
        let dx0 = x0[[1]] - x0[[0]];
        let dx1 = x1[[1]] - x1[[0]];
        let dx2 = x2[[1]] - x2[[0]];
        let dx3 = x3[[1]] - x3[[0]];
        let xspace = Xspace4D {
            x0: [x0[[0]], x1[[0]], x2[[0]], x3[[0]]],
            dx: [dx0, dx1, dx2, dx3],
            n: [x0.len(), x1.len(), x2.len(), x3.len()],
            grid: [x0, x1, x2, x3],
        };

        let p = Pspace4D::init(&xspace);
        let dpsi_d0 = None;
        let dpsi_d1 = None;
        let dpsi_d2 = None;
        let dpsi_d3 = None;
        Self {
            psi,
            x: xspace,
            p,
            dpsi_d0,
            dpsi_d1,
            dpsi_d2,
            dpsi_d3,
            representation: Representation::Position,
        }
    }
}
//============================================================================================
/// спектральные производные

pub fn fft_dpsi_d0(psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
    let mut fft_maker = FftMaker4D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_3d, p0_point)| {
            multizip((psi_3d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_2d, _p1_point)| {
                    multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[2].iter())).for_each(
                        |(mut psi_1d, _p2_point)| {
                            multizip((psi_1d.iter_mut(), p.grid[3].iter())).for_each(
                                |(elem, _p3_point)| {
                                    *elem *= I * p0_point;
                                },
                            );
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}

pub fn fft_dpsi_d1(psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
    let mut fft_maker = FftMaker4D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_3d, _p0_point)| {
            multizip((psi_3d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_2d, p1_point)| {
                    multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[2].iter())).for_each(
                        |(mut psi_1d, _p2_point)| {
                            multizip((psi_1d.iter_mut(), p.grid[3].iter())).for_each(
                                |(elem, _p3_point)| {
                                    *elem *= I * p1_point;
                                },
                            );
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}

pub fn fft_dpsi_d2(psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
    let mut fft_maker = FftMaker4D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_3d, _p0_point)| {
            multizip((psi_3d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_2d, _p1_point)| {
                    multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[2].iter())).for_each(
                        |(mut psi_1d, p2_point)| {
                            multizip((psi_1d.iter_mut(), p.grid[3].iter())).for_each(
                                |(elem, _p3_point)| {
                                    *elem *= I * p2_point;
                                },
                            );
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}

pub fn fft_dpsi_d3(psi: &mut Array4<C>, x: &Xspace4D, p: &Pspace4D) {
    let mut fft_maker = FftMaker4D::new(&x.n);
    fft_maker.modify(psi, x, p);
    fft_maker.fft(psi);

    multizip((psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_3d, _p0_point)| {
            multizip((psi_3d.axis_iter_mut(Axis(0)), p.grid[1].iter())).for_each(
                |(mut psi_2d, _p1_point)| {
                    multizip((psi_2d.axis_iter_mut(Axis(0)), p.grid[2].iter())).for_each(
                        |(mut psi_1d, _p2_point)| {
                            multizip((psi_1d.iter_mut(), p.grid[3].iter())).for_each(
                                |(elem, p3_point)| {
                                    *elem *= I * p3_point;
                                },
                            );
                        },
                    );
                },
            );
        });

    fft_maker.ifft(psi);
    fft_maker.demodify(psi, x, p);
}
