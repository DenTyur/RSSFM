use crate::config;
use crate::field;
use crate::gauge;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use config::{Compl, Float, I};
use field::Field2D;
use gauge::{LenthGauge, VelocityGauge};
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use plotters::prelude::*;
use potentials::{br_1e2d_external, AtomicPotential};
use rayon::prelude::*;
use std::f32::consts::PI;
use wave_function::WaveFunction;

pub struct PML<'a> {
    pub sigma0: Float,
    pub omega: Float,
    pub width: Float, //ширина
    pub n: i32,
    pub x: &'a Xspace,
    pub sigma: [Array<Float, Ix1>; 2],
}

impl<'a> PML<'a> {
    pub fn new(sigma0: Float, omega: Float, width: Float, n: i32, x: &'a Xspace) -> Self {
        let x_left = x.grid[0][0] + width;
        let x_right = x.grid[0][x.n[0] - 1] - width;
        let sigma_in_point_left = |x: Float| sigma0 * ((x_left - x).abs() / width).powi(n);
        let sigma_in_point_right = |x: Float| sigma0 * ((x - x_right).abs() / width).powi(n);
        let mut sigma_x: Array<Float, Ix1> = Array::zeros(x.n[0]);
        sigma_x
            .iter_mut()
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(sigma_elem, x_point)| {
                if *x_point < x_left {
                    *sigma_elem = sigma_in_point_left(*x_point);
                }
                if *x_point > x_right {
                    *sigma_elem = sigma_in_point_right(*x_point);
                }
            });
        let mut sigma_y: Array<Float, Ix1> = Array::zeros(x.n[1]);
        sigma_y
            .iter_mut()
            .zip(x.grid[1].iter())
            .par_bridge()
            .for_each(|(sigma_elem, y_point)| {
                if *y_point < x_left {
                    *sigma_elem = sigma_in_point_left(*y_point);
                }
                if *y_point > x_right {
                    *sigma_elem = sigma_in_point_right(*y_point);
                }
            });
        let sigma = [sigma_x, sigma_y];

        Self {
            omega,
            width,
            n,
            x,
            sigma0,
            sigma,
        }
    }
}
