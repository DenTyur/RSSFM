extern crate fstrings;

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;

pub struct Point2e1D {
    pub x1: f32,
    pub x2: f32,
}

#[derive(Debug, Clone)]
pub struct Tspace {
    // параметры временной сетки
    pub t0: f32,
    pub dt: f32,
    pub n_steps: usize,
    pub nt: usize,
    pub current: f32,
    pub grid: Array<f32, Ix1>,
}

impl Tspace {
    pub fn new(t0: f32, dt: f32, n_steps: usize, nt: usize) -> Self {
        Self {
            t0,
            dt,
            n_steps,
            nt,
            current: t0,
            // костыль
            grid: Array::linspace(t0, t0 + dt * n_steps as f32 * (nt - 1) as f32, nt),
        }
    }

    pub fn t_step(&self) -> f32 {
        // возвращает временной шаг срезов волновой функции
        self.dt * self.n_steps as f32
    }

    pub fn last(&self) -> f32 {
        // возвращает последний элемент сетки временных срезов
        self.t0 + self.t_step() * (self.nt - 1) as f32
    }

    pub fn get_grid(&self) -> Array<f32, Ix1> {
        // возвращает временную сетку срезов
        Array::linspace(self.t0, self.last(), self.nt)
    }

    // Сохраняет временную сетку в файл
    pub fn save_grid(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.grid.write_npy(writer)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Xspace {
    // размерность пространства
    pub dim: usize,
    // параметры координатной сетки
    pub x0: Vec<f32>,
    pub dx: Vec<f32>,
    pub n: Vec<usize>,
    // сетка
    pub grid: Vec<Array<f32, Ix1>>,
}

impl Xspace {
    pub fn new(x0: Vec<f32>, dx: Vec<f32>, n: Vec<usize>) -> Self {
        // Как записать assert(x0.len()==dx.len()==n.len())?
        assert_eq!(x0.len(), dx.len(), "Dimension Error");
        assert_eq!(n.len(), dx.len(), "Dimension Error");
        Self {
            dim: x0.len(),
            x0: x0.clone(), // костыль!
            dx: dx.clone(),
            n: n.clone(),
            grid: (0..x0.len()) // тоже костыль! как это сделать через функцию?
                .into_iter()
                .map(|i| Array::linspace(x0[i], x0[i] + dx[i] * (n[i] - 1) as f32, n[i]))
                // переписать через x0 + dx*arrange(N)
                .collect(),
        }
    }
    pub fn point(&self, index: (usize, usize)) -> Point2e1D {
        Point2e1D {
            x1: self.grid[0][[index.0]],
            x2: self.grid[1][[index.1]],
        }
    }

    pub fn load(dir_path: &str, dim: usize) -> Self {
        // Загружает массивы координат из файлов.
        //
        // dir_path - путь к директории с координатными массивами. Массивы
        // в этой директории должны называться: x0.png, x1.png, x2.png и так
        // далее в зависимости от размерности пространства.
        // Всего таких массивов в этой директории dir_path должо быть dim штук.
        //
        // dim - размерность пространства.

        let mut x: Vec<Array<f32, Ix1>> = Vec::new();
        let mut x0: Vec<f32> = Vec::new();
        let mut dx: Vec<f32> = Vec::new();
        let mut n: Vec<usize> = Vec::new();

        for i in 0..dim {
            let x_path = String::from(dir_path) + f!("/x{i}.npy").as_str();
            let reader = File::open(x_path).unwrap();
            x.push(Array1::<f32>::read_npy(reader).unwrap());
            x0.push(x[i][[0]]);
            dx.push(x[i][[1]] - x[i][[0]]);
            n.push(x[i].len());
        }
        Self {
            dim: dim,
            x0: x0,
            dx: dx,
            n: n,
            grid: x,
        }
    }

    pub fn save(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        // Сохраняет массивы координат в файлы
        //
        // dir_path - путь к папке, в которую будут сохранены массивы.
        //
        // Массивы сохраняются с названиями x0.png, x1.png, x2.png и так
        // далее в зависимости от размерности пространства.

        for i in 0..self.dim {
            let x_path = String::from(dir_path) + f!("/x{i}.npy").as_str();
            let writer = BufWriter::new(File::create(x_path)?);
            self.grid[i].write_npy(writer)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Pspace {
    // размерность пространства
    pub dim: usize,
    // параметры импульсной сетки
    pub p0: Vec<f32>,
    pub dp: Vec<f32>,
    pub n: Vec<usize>,
    // сетка
    pub grid: Vec<Array<f32, Ix1>>,
}

impl Pspace {
    pub fn init(x: &Xspace) -> Self {
        let p0 = x.dx.iter().map(|&dx| -PI / dx).collect::<Vec<_>>();
        let dp =
            x.n.iter()
                .zip(x.dx.iter())
                .map(|(&n, &dx)| 2. * PI / (n as f32 * dx))
                .collect::<Vec<_>>();
        Self {
            dim: x.dim.clone(),
            p0: p0.clone(),
            dp: dp.clone(),
            n: x.n.clone(),
            grid: Vec::from_iter(0..p0.len()) // тоже костыль!
                .iter()
                .map(|&i| Array::linspace(p0[i], p0[i] + dp[i] * (x.n[i] - 1) as f32, x.n[i]))
                .collect(),
        }
    }

    pub fn point_abs_squared(&self, index: (usize, usize)) -> f32 {
        self.grid[0][[index.0]].powi(2) + self.grid[1][[index.1]].powi(2)
    }

    pub fn save(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        // Сохраняет массивы импульсов в файлы
        //
        // dir_path - путь к папке, в которую будут сохранены массивы.
        //
        // Массивы сохраняются с названиями p0.png, p1.png, p2.png и так
        // далее в зависимости от размерности пространства.

        for i in 0..self.dim {
            let p_path = String::from(dir_path) + f!("p{i}.npy").as_str();
            let writer = BufWriter::new(File::create(p_path)?);
            self.grid[i].write_npy(writer)?;
        }
        Ok(())
    }
}
