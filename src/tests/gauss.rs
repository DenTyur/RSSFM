use crate::field;
use crate::flow;
use crate::gauge;
use crate::heatmap;
use crate::logcolormap;
use crate::parameters;
use crate::tsurff;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
use flow::{Circle, Flow, Flux, Square, SurfaceFlow};
use gauge::{LenthGauge, VelocityGauge};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::marker::{Send, Sync};
use std::time::Instant;
use tsurff::Tsurff;
use volkov::{Volkov, VolkovGauge};
use wave_function::{ValueAndSpaceDerivatives, WaveFunction};

type F = f32;
type C = Complex<F>;
const I: C = Complex::I;

/// Аналитическая эволюция гауссового волнового пакета
pub struct Gauss {
    p0: [F; 2],
    sigma: F,
}

impl Gauss {
    pub fn new(p0: [F; 2], sigma: F) -> Self {
        Self { p0, sigma }
    }

    /// Волновая фукнция
    pub fn wf(&self, x: [F; 2], t: F) -> C {
        let v0: [F; 2] = self.p0;
        let e0: F = (self.p0[0].powi(2) + self.p0[1].powi(2)) / 2.0;
        let s = |t: F| self.sigma * (1.0 + I * t / (2.0 * self.sigma.powi(2)));
        let phase = |t: F| {
            I * (self.p0[0] * x[0] + self.p0[1] * x[1])
                - I * e0 * t
                - ((x[0] - v0[0] * t).powi(2) + (x[1] - v0[1] * t).powi(2))
                    / (4.0 * self.sigma * s(t))
        };
        (2.0 * PI).powf(-0.5) / s(t) * phase(t).exp()
    }

    /// Волновая функция массив
    pub fn wf_as_array(&self, x: &Xspace, t: F) -> Array2<C> {
        let mut psi: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_point)| {
                        *psi_elem = self.wf([*x_point, *y_point], t);
                    });
            });
        psi
    }

    /// Производная волновой функции по x
    pub fn dwf_dx(&self, x: [F; 2], t: F) -> C {
        (I * self.p0[0]
            - (x[0] - self.p0[0] * t)
                / (2.0 * self.sigma.powi(2) * (1.0 + I * t / (2.0 * self.sigma.powi(2)))))
            * self.wf(x, t)
    }
    pub fn dwf_dx_as_array(&self, x: &Xspace, t: F) -> Array2<C> {
        let mut psi: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_point)| {
                        *psi_elem = self.dwf_dx([*x_point, *y_point], t);
                    });
            });
        psi
    }
    /// Производная волновой функции по y
    pub fn dwf_dy(&self, x: [F; 2], t: F) -> C {
        (I * self.p0[1]
            - (x[1] - self.p0[1] * t)
                / (2.0 * self.sigma.powi(2) * (1.0 + I * t / (2.0 * self.sigma.powi(2)))))
            * self.wf(x, t)
    }
    pub fn dwf_dy_as_array(&self, x: &Xspace, t: F) -> Array2<C> {
        let mut psi: Array2<C> = Array::zeros((x.n[0], x.n[1]));
        psi.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                psi_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(psi_elem, y_point)| {
                        *psi_elem = self.dwf_dy([*x_point, *y_point], t);
                    });
            });
        psi
    }

    /// Вектор плотности потока вероятности
    pub fn j(&self, x: [F; 2], t: F) -> [C; 2] {
        let wf_sq: C = self.wf(x, t) * self.wf(x, t).conj();
        let jx = wf_sq
            * (self.p0[0]
                + (t * (x[0] - self.p0[0] * t))
                    / (4.0
                        * self.sigma.powi(4)
                        * (1.0 + (t / (2.0 * self.sigma.powi(2))).powi(2))));
        let jy = wf_sq
            * (self.p0[1]
                + (t * (x[1] - self.p0[1] * t))
                    / (4.0
                        * self.sigma.powi(4)
                        * (1.0 + (t / (2.0 * self.sigma.powi(2))).powi(2))));
        [jx, jy]
    }

    /// Импульсное распределение в точке
    pub fn momentum_distribution_in_point(&self, p: [F; 2]) -> F {
        let sigma_p: F = 1.0 / (2.0 * self.sigma);
        1.0 / (2.0 * PI * sigma_p.powi(2))
            * (-((p[0] - self.p0[0]).powi(2) + (p[1] - self.p0[1]).powi(2))
                / (2.0 * sigma_p.powi(2)))
            .exp()
    }

    /// Импульсное распределение массив
    pub fn momentum_distribution_as_array(&self, p: &Pspace) -> Array2<F> {
        let mut momentum_distr: Array2<F> = Array::zeros((p.n[0], p.n[1]));
        momentum_distr
            .axis_iter_mut(Axis(0))
            .zip(p.grid[0].iter())
            .par_bridge()
            .for_each(|(mut distr_row, px_point)| {
                distr_row
                    .iter_mut()
                    .zip(p.grid[1].iter())
                    .for_each(|(distr_elem, py_point)| {
                        *distr_elem = self.momentum_distribution_in_point([*px_point, *py_point]);
                    });
            });
        momentum_distr
    }
    /// график импульсного раапределения
    pub fn plot_momentum_distribution(&self, p: &Pspace, path: &str) {
        //строим импульсное аналитическое импульсное
        let analit_momentum_distr = self.momentum_distribution_as_array(p);
        let (size_x, size_y, size_colorbar) = (500, 500, 60);
        let (colorbar_min, colorbar_max) = (1e-3, 1e-0);

        heatmap::plot_heatmap(
            &p.grid[0],
            &p.grid[1],
            &analit_momentum_distr,
            size_x,
            size_y,
            size_colorbar,
            colorbar_min,
            colorbar_max,
            path,
        );
    }

    pub fn plot_momentum_distribution_log(&self, p: &Pspace, path: &str, colorbar_limits: [F; 2]) {
        //строим импульсное аналитическое импульсное
        let analit_momentum_distr = self.momentum_distribution_as_array(p);
        let (size_x, size_y, size_colorbar) = (500, 500, 60);
        // let (colorbar_min, colorbar_max) = (1e-3, 1e-0);

        logcolormap::plot_heatmap_logscale(
            &analit_momentum_distr,
            &p.grid[0],
            &p.grid[1],
            (colorbar_limits[0], colorbar_limits[1]),
            path,
        );
    }
}
///спектральная производная
pub fn fft_dpsi_dx(psi: &mut Array2<C>, x: &Xspace, p: &Pspace) {
    use super::super::evolution::FftMaker2d;
    use itertools::multizip;
    let mut fft = FftMaker2d::new(&x.n);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // модифицируем psi
                    *psi_elem *= x.dx[0] * x.dx[1] / (2. * PI)
                        * (-I * (p.p0[0] * x_point + p.p0[1] * *y_point)).exp();
                });
        });
    fft.fft(psi);
    psi.axis_iter_mut(Axis(0))
        .zip(p.grid[0].iter())
        .par_bridge()
        .for_each(|(mut distr_row, px_point)| {
            distr_row
                .iter_mut()
                .zip(p.grid[1].iter())
                .for_each(|(distr_elem, py_point)| {
                    *distr_elem *= I * px_point;
                });
        });
    fft.ifft(psi);
    multizip((psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            multizip((psi_row.iter_mut(), x.grid[1].iter()))
                // соединяем второй индекс psi с y
                .for_each(|(psi_elem, y_point)| {
                    // демодифицируем psi
                    *psi_elem *= (2. * PI) / (x.dx[0] * x.dx[1])
                        * (I * (p.p0[0] * x_point + p.p0[1] * y_point)).exp();
                });
        });
}

#[test]
fn ssfm_gauss() {
    use crate::evolution::SSFM;
    use crate::macros::measure_time;

    let mut t = Tspace::new(0., 0.2, 1, 100);
    let dx: F = 0.5;
    let n: usize = 350;
    let x = Xspace::new(vec![-80.0, -80.0], vec![dx, dx], vec![n, n]);
    let p = Pspace::init(&x);

    let field = Field2D {
        amplitude: 0.0,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let gauge = VelocityGauge::new(&field);

    fn potential(_x: F, _y: F) -> F {
        0.0
    }

    fn absorbing_potential(x: F, y: F) -> C {
        let r0: F = 20.0;
        let alpha: F = 0.02;
        let r: F = (x.powi(2) + y.powi(2)).sqrt();
        if r > r0 {
            -I * (r - r0).abs() * alpha
        } else {
            C::new(0.0, 0.0)
        }
    }

    let mut ssfm = SSFM::new(&gauge, &x, &p, potential, absorbing_potential);

    let gauss = Gauss::new([0.0, 3.0], 1.0);
    gauss.plot_momentum_distribution_log(
        &p,
        "src/tests/out/gauss/analit_momemtum_disrt.png",
        [1e-5, 1.0],
    );
    let mut psi = WaveFunction::new(gauss.wf_as_array(&x, 0.0), &x);

    // создаем структуру для потока вероятности через поверхность
    let surface = Square::new(10.0, &x);
    let mut flow = Flow::new(&gauge, &surface);
    // t-surff
    let mut tsurff = Tsurff::new(&gauge, &surface, &x, &p, None);

    let total_time = Instant::now();
    for i in 0..t.nt {
        let time_step = Instant::now();
        println!(
            "STEP {}/{}, t.current={:.5}, norm = {}, prob_in_box = {}",
            i,
            t.nt,
            t.current,
            psi.norm(),
            psi.prob_in_numerical_box(),
        );
        //============================================================
        //                       SSFM
        //============================================================
        measure_time!("SSFM", {
            ssfm.time_step_evol(&mut psi, &mut t, None, None);
        });
        // график волновой функции
        psi.plot_log(
            format!("src/tests/out/gauss/psi_x/psi_x_t_{i}.png").as_str(),
            [1e-8, 1.0],
        );
        //обновление производных
        measure_time!("update_deriv", {
            psi.update_derivatives();
        });
        //============================================================
        //                       t-SURFF
        //============================================================
        measure_time!("tsurff", {
            tsurff.time_integration_step(&psi, &t);
            // if i % 10 == 0 {
            tsurff.plot_log(
                format!("src/tests/out/gauss/tsurff/tsurff_{i}.png").as_str(),
                [1e-5, 1.0],
            );
            // }
        });
        //============================================================
        //                       Flow
        //============================================================
        measure_time!("flow_time", {
            flow.add_instance_flow(&psi, t.current);
        });

        //============================================================
        println!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )
    }

    flow.plot_flow("src/tests/out/gauss/flow_graph.png");
    let total_flow = flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
}

#[test]
fn dgauss_dx() {
    let x_dir_path = "src/arrays_saved";
    let psi_path = "src/arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.1, 1, 150);

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 2);
    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);

    //гауссов пакет
    let gauss = Gauss::new([4.0, 0.0], 1.0);

    let t: F = 0.5;
    let mut psi = gauss.wf_as_array(&x, t);
    let ind = [x.n[0] / 2, x.n[1] / 2];

    let ind_plus = [x.n[0] / 2 + 1, x.n[1] / 2];
    let ind_minus = [x.n[0] / 2 - 1, x.n[1] / 2];

    let point = [x.grid[0][ind[0]], x.grid[1][ind[1]]];
    let point_plus = [x.grid[0][ind_plus[0]], x.grid[1][ind_plus[1]]];
    let point_minus = [x.grid[0][ind_minus[0]], x.grid[1][ind_minus[1]]];
    let dpsi_dx_analit = gauss.dwf_dx(point, t);

    let dpsi_dx = (psi[ind_plus] - psi[ind_minus]) / (1.0 * x.dx[0]);
    println!("dpsi_dx = {}", dpsi_dx);
    println!("dpsi_dx_analit = {}", dpsi_dx_analit);

    println!("psi_plus={}", psi[ind_plus]);
    println!("psi_minus={}", psi[ind_minus]);

    println!("psi_plus_analit={}", gauss.wf([0.5, 0.0], t));
    println!("psi_minus_analit={}", gauss.wf([-0.5, 0.0], t));

    fft_dpsi_dx(&mut psi, &x, &p);
    println!("fft_dpsi_dx={}", psi[ind]);
}

#[test]
/// Шеститочечная производная
fn d_dx_6() {
    use ndarray::{Array1, Array2};
    use num_complex::{Complex32, ComplexFloat};
    use std::f32::consts::PI;

    // Параметры гауссова волнового пакета
    const X0: f32 = 0.0; // Начальное положение по x
    const Y0: f32 = 0.0; // Начальное положение по y
    const SIGMA_X: f32 = 1.0; // Ширина пакета по x
    const SIGMA_Y: f32 = 1.0; // Ширина пакета по y
    const KX: f32 = 2.0; // Волновое число по x
    const KY: f32 = 0.0; // Волновое число по y

    /// Аналитическая форма гауссова волнового пакета в момент времени t=0
    fn gaussian_wave_packet(x: f32, y: f32) -> Complex32 {
        // let exponent =
        //     -0.5 * (((x - X0).powi(2) / SIGMA_X.powi(2)) + ((y - Y0).powi(2) / SIGMA_Y.powi(2)));
        // let envelope = (SIGMA_X * SIGMA_Y * PI).sqrt().recip();
        // let phase = (KX * x + KY * y).mul_add(1.0, 0.0); // e^{i(kx x + ky y)}
        // Complex32::new(envelope * exponent.exp(), 0.0) * Complex32::cis(phase)
        let gauss = Gauss::new([4.0, 0.0], 1.0);
        gauss.wf([x, y], 0.0)
    }

    /// Аналитическая производная гауссова пакета по x
    fn gaussian_wave_packet_derivative_x(x: f32, y: f32) -> Complex32 {
        // let psi = gaussian_wave_packet(x, y);
        // let term1 = -(x - X0) / SIGMA_X.powi(2);
        // let term2 = KX * Complex32::i();
        // psi * (term1 + term2)
        let gauss = Gauss::new([4.0, 0.0], 1.0);
        gauss.dwf_dx([x, y], 0.0)
    }

    /// Численная производная по x (центральная разность)
    fn numerical_derivative_x<F>(f: F, x: f32, y: f32, h: f32) -> Complex32
    where
        F: Fn(f32, f32) -> Complex32,
    {
        //     (-f(x + 2.0 * h, y) + 8.0 * f(x + h, y) - 8.0 * f(x - h, y) + f(x - 2.0 * h, y))
        //         / (12.0 * h)
        (-f(x + 3.0 * h, y) + 9.0 * f(x + 2.0 * h, y) - 45.0 * f(x + h, y) + 45.0 * f(x - h, y)
            - 9.0 * f(x - 2.0 * h, y)
            + f(x - 3.0 * h, y))
            / (60.0 * h)
    }

    // Точка, в которой вычисляем производную
    let x = 0.0;
    let y = 0.0;
    let h = 0.5; // Шаг для численного дифференцирования

    // Аналитическая производная
    let analytic_deriv = gaussian_wave_packet_derivative_x(x, y);
    println!(
        "Аналитическая производная в точке ({}, {}):\n{:.6}",
        x, y, analytic_deriv
    );

    // Численная производная
    let numeric_deriv = numerical_derivative_x(gaussian_wave_packet, x, y, h);
    println!(
        "Численная производная в точке ({}, {}):\n{:.6}",
        x, y, numeric_deriv
    );

    // Разница между аналитическим и численным результатом
    let diff = (analytic_deriv - numeric_deriv).norm();
    println!("Разница между методами: {:.6e}", diff);
}
