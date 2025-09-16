use crate::config::{C, F, I};
use crate::dim2::{space::Xspace2D, wave_function::WaveFunction2D};
use crate::dim4::fft_maker::FftMaker4D;
use crate::dim4::space::{Pspace4D, Xspace4D};
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

pub fn integrate_y1y2(psi: &Array4<C>, y1: &Array1<F>, y2: &Array1<F>, center_cut: F) -> Array2<F> {
    let shape = psi.shape();
    let nx1 = shape[0];
    let ny1 = shape[1];
    let nx2 = shape[2];
    let ny2 = shape[3];

    // Предварительно вычисляем маски для y1 и y2
    let y1_mask: Vec<bool> = y1.iter().map(|&y| y.abs() > center_cut).collect();
    let y2_mask: Vec<bool> = y2.iter().map(|&y| y.abs() > center_cut).collect();

    // Шаги сеток
    let dy1: F = y1[[1]] - y1[[0]];
    let dy2: F = y2[[1]] - y2[[0]];

    // Создаем массив и параллельно заполняем его
    let mut result: Array2<F> = Array2::zeros((nx1, nx2));

    result
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(ix1, mut row)| {
            for ix2 in 0..nx2 {
                let mut sum: F = 0.0;

                for iy1 in 0..ny1 {
                    if y1_mask[iy1] {
                        for iy2 in 0..ny2 {
                            if y2_mask[iy2] {
                                let psi_val = psi[[ix1, iy1, ix2, iy2]];
                                sum += (psi_val.im.powi(2) + psi_val.re.powi(2)) * dy1 * dy2;
                            }
                        }
                    }
                }

                row[ix2] = sum;
            }
        });

    result
}
