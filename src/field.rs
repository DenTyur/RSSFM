use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use plotly::common::{Marker, Mode, Title};
use plotly::layout::{Axis, Layout};
use plotly::{Plot, Scatter};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;

use crate::parameters::{self, Tspace};
use parameters::Xspace;

pub struct Field2D {
    pub amplitude: f32,
    pub omega: f32,
    pub N: f32,
    pub x_envelop: f32,
}

impl Field2D {
    pub fn new(amplitude: f32, omega: f32, N: f32, x_envelop: f32) -> Self {
        Self {
            amplitude,
            omega,
            N,
            x_envelop,
        }
    }
    pub fn electric_field_time_dependence(&self, t: f32) -> [f32; 2] {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        let mut electric_field: [f32; 2] = [0., 0.];

        if 2. * PI * self.N / self.omega - t > 0. {
            electric_field[0] = -self.amplitude
                * f32::sin(self.omega * t / (2. * self.N)).powi(2)
                * f32::sin(self.omega * t);
            electric_field[1] = -self.amplitude
                * f32::sin(self.omega * t / (2. * self.N)).powi(2)
                * f32::cos(self.omega * t);
        }
        electric_field
    }

    pub fn electric_field(&self, t: f32, x: f32, y: f32) -> [f32; 2] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [
            self.electric_field_time_dependence(t)[0],
            self.electric_field_time_dependence(t)[1],
        ]
    }

    pub fn scalar_potential(&self, t: f32, x: f32, y: f32) -> f32 {
        -self.electric_field_time_dependence(t)[0] * x
            - self.electric_field_time_dependence(t)[1] * y
    }

    // Векторный потенциал
    pub fn vec_pot(&self, t: f32) -> [f32; 2] {
        let mut vec_pot_x: f32 = 0.0;
        let mut vec_pot_y: f32 = 0.0;
        let compute_vec_pot_x = |tau: f32| {
            self.amplitude
                * (-1.0
                    + f32::cos(tau)
                        * (1.0 - self.N.powi(2) + self.N.powi(2) * f32::cos(tau / self.N))
                    + self.N * f32::sin(tau) * f32::sin(tau / self.N))
                / (2.0 * self.omega * (-1.0 + self.N.powi(2)))
        };
        let compute_vec_pot_y = |tau: f32| {
            self.amplitude
                * (f32::sin(tau)
                    * (-1.0 + self.N.powi(2) - self.N.powi(2) * f32::cos(tau / self.N))
                    + self.N * f32::cos(tau) * f32::sin(tau / self.N))
                / (2.0 * self.omega * (-1.0 + self.N.powi(2)))
        };

        let tau: f32 = self.omega * t;
        let period: f32 = 2. * PI * self.N / self.omega;
        if t < period {
            vec_pot_x = compute_vec_pot_x(tau);
            vec_pot_y = compute_vec_pot_y(tau);
        } else {
            vec_pot_x = compute_vec_pot_x(period);
            vec_pot_y = compute_vec_pot_y(period);
        }
        [vec_pot_x, vec_pot_y]

        // if 2. * PI * self.N / self.omega - t > 0. {
        //     vec_pot[0] = self.amplitude / self.omega
        //         * f32::sin(self.omega * t / (2. * self.N)).powi(2)
        //         * f32::sin(self.omega * t);
        //     vec_pot[1] = self.amplitude / self.omega
        //         * f32::sin(self.omega * t / (2. * self.N)).powi(2)
        //         * f32::cos(self.omega * t);
        // }
        // vec_pot
    }

    // интеграл от векторного потенциала
    pub fn a(&self, t: f32) -> [f32; 2] {
        let tau: f32 = self.omega * t;
        let mut a: [f32; 2] = [0.0, 0.0];

        // a[0] = self.amplitude
        //     * t
        //     * (-1.0
        //         + f32::cos(tau) * (1.0 - self.N.powi(2) + self.N.powi(2) * f32::cos(tau / self.N))
        //         + self.N * f32::sin(tau) * f32::sin(tau / self.N))
        //     / (2.0 * (-1.0 + self.N.powi(2)) * self.omega);
        a[0] = self.amplitude
            * (-(-1.0 + self.N.powi(2)) * tau
                + f32::sin(tau)
                    * (-(-1.0 + self.N.powi(2)).powi(2)
                        + (self.N.powi(2) + self.N.powi(4)) * f32::cos(tau / self.N))
                - 2.0 * self.N.powi(3) * f32::cos(tau) * f32::sin(tau / self.N))
            / (2.0 * (self.N.powi(2) - 1.0).powi(2) * self.omega.powi(2));

        a[1] = self.amplitude
            * (1.0 - 3.0 * self.N.powi(2)
                + f32::cos(tau)
                    * (-(-1.0 + self.N.powi(2)).powi(2)
                        + (self.N.powi(2) + self.N.powi(4)) * f32::cos(tau / self.N))
                + 2.0 * self.N.powi(3) * f32::sin(tau) * f32::sin(tau / self.N))
            / (2.0 * (self.N.powi(2) - 1.0).powi(2) * self.omega.powi(2));

        a
    }

    // интеграл от квадрата векторного потенциала
    pub fn b(&self, t: f32) -> f32 {
        let tau: f32 = self.omega * t;

        let term1: f32 =
            self.amplitude.powi(2) / (16.0 * (-1.0 + self.N.powi(2)).powi(3) * self.omega.powi(3));
        let term2: f32 = 8.0
            * (-(-1.0 + self.N.powi(2)).powi(2)
                + (self.N.powi(2) + self.N.powi(4)) * f32::cos(tau / self.N))
            * f32::sin(tau);
        let term3: f32 = 8.0
            * self.N.powi(3)
            * ((-1.0 + self.N.powi(2)).powi(2) - 2.0 * f32::cos(tau))
            * f32::sin(tau / self.N);
        let term4: f32 = (-1.0 + self.N.powi(2))
            * (2.0 * (4.0 - 3.0 * self.N.powi(2) + 3.0 * self.N.powi(4)) * tau
                + self.N.powi(3) * (-1.0 + self.N.powi(2)) * f32::sin(2.0 * tau / self.N));
        term1 * (-term2 - term3 + term4)
    }
}

#[test]
fn test_plot_vec_pot() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut vpx: Vec<f32> = Vec::new();
    let mut vpy: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        vpx.push(field.vec_pot(t.grid[[i]])[0]);
        vpy.push(field.vec_pot(t.grid[[i]])[1]);
    }

    let layout = Layout::new()
        .width(800)
        .height(800)
        .title(Title::from("electric field"));
    let trace = Scatter::new(vpx, vpy);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}

#[test]
fn test_plot_electric_field() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut Ex: Vec<f32> = Vec::new();
    let mut Ey: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        Ex.push(field.electric_field_time_dependence(t.grid[[i]])[0]);
        Ey.push(field.electric_field_time_dependence(t.grid[[i]])[1]);
    }

    let layout = Layout::new()
        .width(800)
        .height(800)
        .title(Title::from("vec_pot"));
    let trace = Scatter::new(Ex, Ey);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}

#[test]
fn test_plot_a() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut ax: Vec<f32> = Vec::new();
    let mut ay: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        ax.push(field.a(t.grid[[i]])[0]);
        ay.push(field.a(t.grid[[i]])[1]);
    }

    let layout = Layout::new().width(800).height(800).title(Title::from("a"));
    let trace = Scatter::new(ax, ay);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}

#[test]
fn test_plot_ax() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut ax: Vec<f32> = Vec::new();
    let mut tvec: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        ax.push(field.a(t.grid[[i]])[0]);
        tvec.push(t.grid[[i]]);
    }

    let layout = Layout::new()
        .width(800)
        .height(800)
        .title(Title::from("ax"));
    let trace = Scatter::new(tvec, ax);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}

#[test]
fn test_plot_ay() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut ay: Vec<f32> = Vec::new();
    let mut tvec: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        ay.push(field.a(t.grid[[i]])[1]);
        tvec.push(t.grid[[i]]);
    }

    let layout = Layout::new()
        .width(800)
        .height(800)
        .title(Title::from("ay"));
    let trace = Scatter::new(tvec, ay);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}

#[test]
fn test_plot_b() {
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    let t = Tspace::new(0.0, 0.2, 10, 300);
    let mut b: Vec<f32> = Vec::new();
    let mut tvec: Vec<f32> = Vec::new();

    for i in 0..t.nt {
        b.push(field.b(t.grid[[i]]));
        tvec.push(t.grid[[i]]);
    }

    let layout = Layout::new().width(800).height(800).title(Title::from("b"));
    let trace = Scatter::new(tvec, b);
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);
    plot.show();
}
