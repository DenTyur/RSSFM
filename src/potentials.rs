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
pub fn br_1e2d_external(x: F, y: F) -> C {
    let c1: F = 0.48;
    let c2: F = 2.0;
    let abs_pot: C = absorbing_potential(x, y);
    -c1 * (-(x / c2).powi(2) - (y / c2).powi(2)).exp() + abs_pot
}

pub fn absorbing_potential(x: F, y: F) -> C {
    let r: F = (x.powi(2) + y.powi(2)).sqrt(); //(x.im.powi(2) + x.re.powi(2) + y.re.powi(2) + y.im.powi(2)).sqrt();
    let r0: F = 50.0;
    let alpha: F = 0.02;
    if r > r0 {
        -I * (r - r0).abs() * alpha
    } else {
        C::new(0.0, 0.0)
    }
}
