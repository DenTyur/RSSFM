use crate::config::F;
use crate::macros::check_path;
use ndarray::prelude::*;
use ndarray_npy::{WriteNpyError, WriteNpyExt};
use std::fs::File;
use std::io::BufWriter;

/// Структура для временной сетки
#[derive(Debug, Clone, Default)]
pub struct Tspace {
    pub t0: F,
    pub dt: F,
    pub n_steps: usize,
    pub nt: usize,
    pub current: F,
    pub grid: Array1<F>,
}

impl Tspace {
    pub fn new(t0: F, dt: F, n_steps: usize, nt: usize) -> Self {
        Self {
            t0,
            dt,
            n_steps,
            nt,
            current: t0,
            grid: Array::linspace(t0, t0 + dt * n_steps as F * (nt - 1) as F, nt),
        }
    }

    /// возвращает временной шаг срезов волновой функции
    pub fn t_step(&self) -> F {
        self.dt * self.n_steps as F
    }

    /// возвращает последний элемент сетки временных срезов
    pub fn last(&self) -> F {
        self.t0 + self.t_step() * (self.nt - 1) as F
    }

    /// возвращает временную сетку срезов
    pub fn get_grid(&self) -> Array<F, Ix1> {
        Array::linspace(self.t0, self.last(), self.nt)
    }

    // Сохраняет временную сетку в файл
    pub fn save_grid(&self, path: &str) -> Result<(), WriteNpyError> {
        check_path!(path);
        let writer = BufWriter::new(File::create(path)?);
        self.grid.write_npy(writer)?;
        Ok(())
    }
}
