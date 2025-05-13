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

#[cfg(test)]
mod tests_scalar_product {
    use std::{slice, time::Instant};

    use super::*;

    #[test]
    fn scalar_product() {
        let root = "/home/denis/Programs/atoms_and_ions/br/4D_br_circular/E0035_w004/noninteract_middle_grid/external2D/RSSFM2D/src/arrays_saved";
        let psi_path = "/home/denis/Programs/atoms_and_ions/br/4D_br_circular/E0035_w004/noninteract_middle_grid/external2D/RSSFM2D/src/arrays_saved/time_evol/psi_x/psi_t_21.npy";
        let x = Xspace::load(root, 2);

        let p = Pspace::init(&x);

        let t = Tspace::new(0., 0.2, 100, 22);

        let psi = WaveFunction::init_from_file(psi_path, &x);

        let mut prob_p: Array2<F> = Array::zeros((p.n[0], p.n[1]));

        let field = Field2D {
            amplitude: 0.038,
            omega: 0.04,
            N: 3.0,
            x_envelop: 30.0001,
        };
        let gauge = LenthGauge::new(&field);

        let get_projector = |px: F, py: F| {
            let mut projector: C = C::new(0.0, 0.0);
            let volkov = Volkov::new(&gauge, [px, py], t.last());
            psi.psi
                .axis_iter(Axis(0))
                .zip(x.grid[0].iter())
                .for_each(|(psi_row, x_point)| {
                    psi_row
                        .iter()
                        .zip(x.grid[1].iter())
                        .for_each(|(psi_elem, y_point)| {
                            // обрезаем связанную середину волновой функции
                            if x_point.powi(2) + y_point.powi(2) > 400.0 {
                                projector += volkov.value([*x_point, *y_point]).conj() * psi_elem;
                            }
                        });
                });
            projector * x.dx[0] * x.dx[1]
        };

        // очень долго считается. придется обрезать :)
        let cut_p = 2.0;
        let ip_min = ((-cut_p - p.grid[0][[0]]) / p.dp[0]).round() as usize;
        let ip_max = ((cut_p - p.grid[0][[0]]) / p.dp[0]).round() as usize;

        let slice_step = 5;
        let mut prob_p_cut =
            prob_p.slice_mut(s![ip_min..ip_max;slice_step, ip_min..ip_max;slice_step]);
        let px_cut = p.grid[0].slice(s![ip_min..ip_max;slice_step]);
        let py_cut = p.grid[1].slice(s![ip_min..ip_max;slice_step]);

        println!("len px, py = {}, {}", px_cut.len(), py_cut.len());
        println!(
            "d px, py = {}, {}",
            px_cut[[1]] - px_cut[[0]],
            py_cut[[1]] - py_cut[[0]]
        );

        multizip((prob_p_cut.axis_iter_mut(Axis(0)), px_cut.iter()))
            .par_bridge()
            .for_each(|(mut proj_row, px)| {
                let time = Instant::now();
                println!("px={}/{}", px, px_cut[px_cut.len() - 1]);
                proj_row
                    .iter_mut()
                    .zip(py_cut.iter())
                    .for_each(|(proj_elem, py)| {
                        let a_p = get_projector(*px, *py);
                        *proj_elem = (a_p.conj() * a_p).re;
                    });

                println!("time = {}", time.elapsed().as_secs_f32());
            });

        fn save2d(arr: Array2<F>, path: &str) {
            let writer = BufWriter::new(File::create(path).unwrap());
            arr.write_npy(writer).unwrap();
        }

        fn save1d(arr: Array1<F>, path: &str) {
            let writer = BufWriter::new(File::create(path).unwrap());
            arr.write_npy(writer).unwrap();
        }
        save2d(prob_p_cut.to_owned(), "tests_out/volkov/prob_p_cut.npy");
        save1d(px_cut.to_owned(), "tests_out/volkov/px_cut.npy");
        save1d(py_cut.to_owned(), "tests_out/volkov/py_cut.npy");

        let (size_x, size_y, size_colorbar) = (500, 500, 60);
        let (colorbar_min, colorbar_max) = (1e-2, 5e-1);
        heatmap::plot_heatmap(
            &px_cut.to_owned(),
            &py_cut.to_owned(),
            &prob_p_cut.to_owned(),
            size_x,
            size_y,
            size_colorbar,
            colorbar_min,
            colorbar_max,
            format!("tests_out/volkov/prob_p_rust.png").as_str(),
        )
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
            format!("volkov_{i}.png").as_str(),
        )
    }
}
