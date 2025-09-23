use crate::config::{F, PI};
use crate::macros::check_path;
use crate::traits::space::Space;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Clone)]
pub struct Xspace1D {
    pub x0: [F; 1],
    pub dx: [F; 1],
    pub n: [usize; 1],
    pub grid: [Array1<F>; 1],
}

impl Xspace1D {
    pub const DIM: usize = 1;
    pub const PREFIX: &'static str = "x";

    pub fn new(x0: [F; Self::DIM], dx: [F; Self::DIM], n: [usize; Self::DIM]) -> Self {
        assert_eq!(x0.len(), dx.len(), "Dimension Error");
        assert_eq!(n.len(), dx.len(), "Dimension Error");
        let mut grid: [Array1<F>; 1] = [Array1::default(0)];
        grid[0] = Array::linspace(x0[0], x0[0] + dx[0] * (n[0] - 1) as F, n[0]);
        Self { x0, dx, n, grid }
    }
}

impl Space<1> for Xspace1D {
    fn point(&self, index: [usize; Self::DIM]) -> [F; Self::DIM] {
        [self.grid[0][index[0]]]
    }
    fn load_from_npy(dir_path: &str) -> Self {
        let mut grid: [Array1<F>; Self::DIM] = [Array1::default(0)];
        let mut x0: [F; Self::DIM] = [0.0];
        let mut dx: [F; Self::DIM] = [0.0];
        let mut n: [usize; Self::DIM] = [0];

        let i = 0_usize;
        let x_path = String::from(dir_path) + "/" + Self::PREFIX + format!("{i}.npy").as_str();
        dbg!(&x_path);
        let reader = File::open(x_path).unwrap();
        grid[i] = Array1::read_npy(reader).unwrap();
        x0[i] = grid[i][[0]];
        dx[i] = grid[i][[1]] - grid[i][[0]];
        n[i] = grid[i].len();

        Self { x0, dx, n, grid }
    }

    fn save_as_npy(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        let i = 0_usize;
        let x_path = String::from(dir_path) + "/" + Self::PREFIX + format!("{i}.npy").as_str();
        let writer = BufWriter::new(File::create(x_path)?);
        self.grid[i].write_npy(writer)?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Pspace1D {
    pub p0: [F; 1],
    pub dp: [F; 1],
    pub n: [usize; 1],
    pub grid: [Array1<F>; 1],
}

impl Pspace1D {
    pub const DIM: usize = 1;
    pub const PREFIX: &'static str = "p";

    pub fn init(x: &Xspace1D) -> Self {
        let mut p0: [F; Self::DIM] = [0.0];
        let mut dp: [F; Self::DIM] = [0.0];
        let mut n: [usize; Self::DIM] = [0];
        for i in 0..Self::DIM {
            p0[i] = -PI / x.dx[i];
            n[i] = x.n[i];
            dp[i] = 2.0 * PI / (n[i] as F * x.dx[i])
        }
        let mut grid: [Array1<F>; 1] = [Array1::default(0)];
        for i in 0..Self::DIM {
            grid[i] = Array::linspace(p0[i], p0[i] + dp[i] * (n[i] - 1) as F, n[i]);
        }

        Self { p0, dp, n, grid }
    }

    pub fn save_as_npy(&self, dir_path: &str) -> Result<(), WriteNpyError> {
        check_path!(dir_path);
        for i in 0..Self::DIM {
            let p_path = String::from(dir_path) + "/" + Self::PREFIX + format!("{i}.npy").as_str();
            let writer = BufWriter::new(File::create(p_path)?);
            self.grid[i].write_npy(writer)?;
        }
        Ok(())
    }

    fn load_from_npy(dir_path: &str) -> Self {
        let mut grid: [Array1<F>; Self::DIM] = [Array1::default(0)];
        let mut p0: [F; Self::DIM] = [0.0];
        let mut dp: [F; Self::DIM] = [0.0];
        let mut n: [usize; Self::DIM] = [0];

        for i in 0..Self::DIM {
            let p_path = String::from(dir_path) + "/" + Self::PREFIX + format!("{i}.npy").as_str();
            let reader = File::open(p_path).unwrap();
            grid[i] = Array1::<F>::read_npy(reader).unwrap();
            p0[i] = grid[i][[0]];
            dp[i] = grid[i][[1]] - grid[i][[0]];
            n[i] = grid[i].len();
        }
        Self { p0, dp, n, grid }
    }
}
