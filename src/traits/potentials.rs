use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

use crate::field;
use crate::parameters;
use parameters::Xspace;

type F = f32;
type C = Complex<f32>;
const I: C = Complex::I;

// pub struct Point1D {
//     pub x: f32,
// }
// //
// pub struct Potential1D {
//     pub atomic_potential: AtomicPotential1D,
//     pub field: Field1D,
// }
// impl Potential1D {
//     pub fn total_potential(&self, index: [usize; 2], t: f32, x: Array<f32, Ix1>) -> Complex<f32> {
//         let j = Complex::I;
//         let uf: f32 = self.field.electric_field_potential(t);
//         self.atomic_potential[index] - uf.x
//     }
// }
//
// pub struct AtomicPotential1D {
//     pub potential: Array<Complex<f32>, Ix1>,
// }
// impl AtomicPotential1D {
//     // Загружает потенциал из файла. path - путь к массиву потенциала.
//     pub fn load_atomic_potential(path: &str) -> Self {
//         let reader = File::open(path).unwrap();
//         Self {
//             potential: Array::<Complex<f32>, Ix1>::read_npy(reader).unwrap(),
//         }
//     }
// }
//==============================

pub struct AtomicPotential<'a> {
    pub potential: Array<Complex<f32>, Ix2>,
    x: &'a Xspace,
}

impl<'a> AtomicPotential<'a> {
    pub fn init_oscillator_2d(x: &'a Xspace) -> Self {
        // плохо написано. Лучше не создавать промежуточное u.
        let mut atomic_potential: Array<Complex<f32>, Ix2> = Array::zeros((x.n[0], x.n[1]));
        let j: Complex<f32> = Complex::I;
        atomic_potential
            .axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut u_row, x_i)| {
                u_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(u_elem, y_j)| {
                        *u_elem = 0.5 * (x_i.powi(2) + y_j.powi(2)) + 0. * j;
                    })
            });
        Self {
            potential: atomic_potential,
            x,
        }
    }

    pub fn save_potential(&self, path: &str) -> Result<(), WriteNpyError> {
        let writer = BufWriter::new(File::create(path)?);
        self.potential.write_npy(writer)?;
        Ok(())
    }

    pub fn init_from_file(atomic_potential_path: &str, x: &'a Xspace) -> Self {
        // Загружает потенциал из файла.

        let reader = File::open(atomic_potential_path).unwrap();
        Self {
            potential: Array::<Complex<f32>, Ix2>::read_npy(reader).unwrap(),
            x,
        }
    }
}

//====================================================================================
//                       Аналитические атомные потенциалы
//====================================================================================

// потенциал отрицательного иона брома. Внешний электрон. Двумерный.
pub fn br_1e2d_external(x: F, y: F) -> F {
    let c1: F = 0.48;
    let c2: F = 2.0;
    let r: F = (x.powi(2) + y.powi(2)).sqrt();
    // let pot = |r: F| -c1 * (-(x / c2).powi(2) - (y / c2).powi(2)).exp();
    let pot = |r: F| -c1 * (-(r / c2).powi(2)).exp();
    if r < 20.0 {
        pot(r)
    } else {
        pot(20.0)
    }
}

//====================================================================================
//                       Комплексные поглощающие потенциалы
//====================================================================================
pub fn absorbing_potential(x: F, y: F) -> C {
    let r: F = (x.powi(2) + y.powi(2)).sqrt();
    let r0: F = 35.0;
    let alpha: F = 0.02;
    if r > r0 {
        -I * (r - r0) * alpha * (1.0 - (-0.5 * (r - r0)).exp())
    } else {
        C::new(0.0, 0.0)
    }
}

pub fn absorbing_potential_linear(x: F, y: F) -> C {
    let r: F = (x.powi(2) + y.powi(2)).sqrt();
    let r0: F = 50.0;
    let alpha: F = 0.02;
    if r > r0 {
        -I * (r - r0).abs() * alpha
    } else {
        C::new(0.0, 0.0)
    }
}

#[test]
fn cap_linear() {
    use super::*;
    use plotters::prelude::*;
    let x_dir_path = "src/arrays_saved";
    let x = Xspace::load(x_dir_path, 2);

    let mut cap_arr: Array<F, Ix2> = Array::zeros((x.n[0], x.n[1]));
    let mut colorbar_max: F = 0.0;
    let mut colorbar_min: F = 1e5;
    cap_arr
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .for_each(|(mut cap_row, x_point)| {
            cap_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(cap_elem, y_point)| {
                    *cap_elem = absorbing_potential_linear(*x_point, *y_point).norm();
                    if *cap_elem > colorbar_max {
                        colorbar_max = *cap_elem;
                    }
                    if *cap_elem < colorbar_min {
                        colorbar_min = *cap_elem;
                    }
                });
        });

    // Создаём график
    let (size_x, size_y, size_colorbar) = (500, 500, 60);

    heatmap::plot_heatmap(
        &x.grid[0],
        &x.grid[1],
        &cap_arr,
        size_x,
        size_y,
        size_colorbar,
        colorbar_min,
        colorbar_max,
        "cap.png",
    )
}

#[test]
fn cap() {
    use super::*;
    use plotters::prelude::*;
    let x_dir_path = "src/arrays_saved";
    let x = Xspace::load(x_dir_path, 2);

    let mut cap_arr: Array<F, Ix2> = Array::zeros((x.n[0], x.n[1]));
    let mut colorbar_max: F = 0.0;
    let mut colorbar_min: F = 1e5;
    cap_arr
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .for_each(|(mut cap_row, x_point)| {
            cap_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(cap_elem, y_point)| {
                    *cap_elem = absorbing_potential(*x_point, *y_point).norm();
                    if *cap_elem > colorbar_max {
                        colorbar_max = *cap_elem;
                    }
                    if *cap_elem < colorbar_min {
                        colorbar_min = *cap_elem;
                    }
                });
        });

    // Создаём график
    let (size_x, size_y, size_colorbar) = (500, 500, 60);

    heatmap::plot_heatmap(
        &x.grid[0],
        &x.grid[1],
        &cap_arr,
        size_x,
        size_y,
        size_colorbar,
        colorbar_min,
        colorbar_max,
        "cap.png",
    )
}
