#![allow(dead_code, non_snake_case, unused_variables, unused_imports)]
use crate::common::{particle::Particle, tspace::Tspace};
use crate::config::{C, F, PI};
use crate::dim2::{
    field::UnipolarPulse1e2d,
    gauge::{LenthGauge2D, VelocityGauge2D},
};
use crate::dim4::{
    fft_maker::FftMaker4D,
    gauge::{LenthGauge4D, VelocityGauge4D},
    space::Xspace4D,
    ssfm::SSFM4D,
    time_fft::TimeFFT,
    wave_function::WaveFunction4D,
    wave_function_processing::probability_density_2d::ProbabilityDensity2D,
};
use crate::measure_time;
use crate::potentials::absorbing_potentials::{
    absorbing_potential_4d, absorbing_potential_4d_asim,
};
use crate::potentials::potentials;
use crate::print_and_log;
use crate::traits::fft_maker::FftMaker;
use crate::traits::{
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use crate::utils::plot_log::plot_log;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::array;
use std::time::Instant;

pub fn create_zeroed_wavefunction(original: &WaveFunction4D, threshold: F) -> WaveFunction4D {
    // Полная копия
    let mut new_wf = WaveFunction4D {
        psi: original.psi.clone(),
        dpsi_d0: original.dpsi_d0.clone(),
        dpsi_d1: original.dpsi_d1.clone(),
        dpsi_d2: original.dpsi_d2.clone(),
        dpsi_d3: original.dpsi_d3.clone(),
        x: original.x.clone(),
        p: original.p.clone(),
        representation: original.representation,
    };

    // Находим граничные индексы для каждой оси
    let idx_0 = find_zeroed_index(&original.x.grid[0], threshold);
    let idx_1 = find_zeroed_index(&original.x.grid[1], threshold);
    let idx_2 = find_zeroed_index(&original.x.grid[2], threshold);
    let idx_3 = find_zeroed_index(&original.x.grid[3], threshold);

    // Зануляем только нужную область
    zero_region(&mut new_wf.psi, idx_0, idx_1, idx_2, idx_3);

    new_wf
}

// Находит индексы, между которыми надо занулять
fn find_zeroed_index(grid: &Array1<F>, threshold: F) -> [usize; 2] {
    let idx_right = grid
        .iter()
        .position(|&x| x >= threshold)
        .unwrap_or(grid.len());

    let idx_left = grid.iter().position(|&x| x >= -threshold).unwrap_or(0);

    [idx_left, idx_right]
}

fn zero_region(
    array: &mut Array4<C>,
    idx_0: [usize; 2],
    idx_1: [usize; 2],
    idx_2: [usize; 2],
    idx_3: [usize; 2],
) {
    // +1 чтобы было включительно
    for i in idx_0[0]..idx_0[1] + 1 {
        for j in idx_1[0]..idx_1[1] + 1 {
            for k in idx_2[0]..idx_2[1] + 1 {
                for l in idx_3[0]..idx_3[1] + 1 {
                    array[[i, j, k, l]] = C::new(0.0, 0.0);
                }
            }
        }
    }
}
