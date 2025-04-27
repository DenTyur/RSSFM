use crate::field;
use crate::gauge;
use crate::heatmap;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field2D;
use gauge::{LenthGauge, VelocityGauge};
use itertools::multizip;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::{br_1e2d_external, AtomicPotential};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use wave_function::{ValueAndSpaceDerivatives, WaveFunction};
// Чем более непонятным становится код, тем он круче! (с)
// Всратость не есть недостаток!

type F = f32;
type C = Complex<f32>;
const I: C = Complex::I;

pub trait VolkovGauge {
    fn compute_phase(&self, x: [F; 2], p: [F; 2], t: F) -> F;
    // множитель, который сносится из экспоненты при дифференцировании по r
    fn deriv_factor(&self, p: [F; 2], t: F) -> [C; 2];
}

impl<'a> VolkovGauge for VelocityGauge<'a> {
    fn compute_phase(&self, x: [F; 2], p: [F; 2], t: F) -> F {
        let p_sq = p[0].powi(2) + p[1].powi(2);
        let a = self.field.a(t);
        //=========================================================
        //     Надо определиться со знаками!
        //=========================================================
        -0.5 * t * p_sq + (p[0] * x[0] + p[1] * x[1]) - (p[0] * a[0] + p[1] * a[1])
    }

    // множитель, который сносится из экспоненты при дифференцировании по r
    fn deriv_factor(&self, p: [F; 2], t: F) -> [C; 2] {
        [I * p[0], I * p[1]]
    }
}

impl<'a> VolkovGauge for LenthGauge<'a> {
    fn compute_phase(&self, x: [F; 2], p: [F; 2], t: F) -> F {
        let p_sq = p[0].powi(2) + p[1].powi(2);
        let vec_pot = self.field.vec_pot(t);
        let a = self.field.a(t);
        let b = self.field.b(t);
        -0.5 * t * p_sq + (p[0] * x[0] + p[1] * x[1]) - (p[0] * vec_pot[0] + p[1] * vec_pot[1])
            + (p[0] * a[0] + p[1] * a[1])
            - 0.5 * b
    }

    // множитель, который сносится из экспоненты при дифференцировании по r
    fn deriv_factor(&self, p: [F; 2], t: F) -> [C; 2] {
        let vec_pot = self.field.vec_pot(t);
        [I * (p[0] - vec_pot[0]), I * (p[1] - vec_pot[1])]
    }
}

pub struct Volkov<'a, G: VolkovGauge> {
    gauge: &'a G,
    pub p: [F; 2],
    pub t: F,
}

impl<'a, G: VolkovGauge> Volkov<'a, G> {
    pub fn new(gauge: &'a G, p: [F; 2], t: F) -> Self {
        Self { gauge, p, t }
    }
}

impl<'a, G: VolkovGauge> ValueAndSpaceDerivatives for Volkov<'a, G> {
    fn deriv(&self, x: [F; 2]) -> [C; 2] {
        let deriv_factor = self.gauge.deriv_factor(self.p, self.t);
        [
            deriv_factor[0] * self.value(x),
            deriv_factor[1] * self.value(x),
        ]
    }

    fn value(&self, x: [F; 2]) -> C {
        let phase = self.gauge.compute_phase(x, self.p, self.t);
        (I * phase).exp() / (2.0 * PI)
    }
}

#[test]
fn volkov_evol() {
    let x_dir_path = "/home/denis/Programs/atoms_and_ions/br/4D_br_circular/E0035_w004/noninteract_big_grid/external2D/RSSFM2D/src/arrays_saved";

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 2);

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 50, 60);
    t.save_grid("/home/denis/Programs/atoms_and_ions/br/4D_br_circular/E0035_w004/noninteract_big_grid/external2D/RSSFM2D/src/arrays_saved/time_evol/volkov/t.npy").unwrap();

    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let gauge = VelocityGauge::new(&field);
    let p = [0.0, 0.2];

    fn save2dc(arr: Array2<C>, path: &str) {
        let writer = BufWriter::new(File::create(path).unwrap());
        arr.write_npy(writer).unwrap();
    }

    let (size_x, size_y, size_colorbar) = (500, 500, 60);
    let (dx, dy) = (1.0, 1.0);
    let x_arr = Array1::range(-80.0, 80.0, dx);
    let y_arr = Array1::range(-80.0, 80.0, dy);
    let (colorbar_min, colorbar_max) = (1e-2, 1e-1);

    for i in 0..t.nt {
        let volkov = Volkov::new(&gauge, p, t.grid[[i]]);
        let mut volkov_arr: Array2<F> = Array::zeros((x_arr.len(), y_arr.len()));
        volkov_arr
            .axis_iter_mut(Axis(0))
            .zip(x_arr.iter())
            .par_bridge()
            .for_each(|(mut psi_row, x)| {
                psi_row
                    .iter_mut()
                    .zip(y_arr.iter())
                    .for_each(|(psi_elem, y)| {
                        let c = volkov.value([*x, *y]);
                        *psi_elem = c.im; //.powi(2) + c.im.powi(2);
                    })
            });

        heatmap::plot_heatmap(
            &x_arr,
            &y_arr,
            &volkov_arr,
            size_x,
            size_y,
            size_colorbar,
            colorbar_min,
            colorbar_max,
            f!("volkov_{i}.png").as_str(),
        )
    }
}
