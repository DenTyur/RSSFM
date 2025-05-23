use crate::config::{F, PI};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

use crate::dim1::space::Xspace1D;

pub struct Field1D {
    pub amplitude: F,
    pub omega: F,
    pub x_envelop: F,
}

impl Field1D {
    pub fn new(amplitude: F, omega: F, x_envelop: F) -> Self {
        Self {
            amplitude,
            omega,
            x_envelop,
        }
    }
    pub fn electric_field_time_dependence(&self, t: F) -> F {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        let mut electric_field: F = 0.;

        if PI / self.omega - t > 0. {
            electric_field = -self.amplitude * F::sin(self.omega * t).powi(2);
        }
        electric_field
    }

    pub fn field_x_envelop(&self, x: F) -> F {
        // Пространственная огибающая электрического поля вдоль каждой из осей.
        F::cos(PI / 2. * x / self.x_envelop).powi(2)
    }

    pub fn integrated_field_x_envelop(&self, x: F) -> F {
        0.5 * x + 0.25 * self.x_envelop * 2. / PI * F::sin(PI * x / self.x_envelop)
    }

    pub fn electric_field(&self, t: F, x: F) -> [F; 1] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [self.electric_field_time_dependence(t) * self.field_x_envelop(x)]
    }

    pub fn vec_pot(&self, t: F) -> [F; 1] {
        let vec_pot: F = 0.0;
        println!("Векторный потенциал не реализован! Заглушка.");
        [vec_pot]
    }

    pub fn scalar_potential(&self, t: F, x: [F; 1]) -> F {
        let x_point = x[0];
        let time_part: F = self.electric_field_time_dependence(t);
        let space_part: F = match x_point {
            x if x <= -self.x_envelop => self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => self.integrated_field_x_envelop(self.x_envelop),
            _ => self.integrated_field_x_envelop(x_point),
        };
        -time_part * space_part
    }

    pub fn potential_as_array(&self, t: F, x: &Xspace1D) -> Array<F, Ix1> {
        // Потенциал электрического поля.
        // В рассматриваемом случае оси независимы (2 одномерных электрона).
        // Поэтому можно интегрировать в 1D.

        let time_part: F = self.electric_field_time_dependence(t);
        let mut space_part: Array<F, Ix1> = x.grid[0].clone();

        // Электрическое поле отлично от нуля в области пространства:
        // -x_envelop < x < x_envelop (*)
        // В этой области пространства производная потенциала этого поля отлична
        // от нуля. За пределами этой области потенциал -- константа, которая равна
        // потенциалу на границах области (*)
        space_part.par_iter_mut().for_each(|elem| match *elem {
            x if x <= -self.x_envelop => *elem = self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => *elem = self.integrated_field_x_envelop(self.x_envelop),
            _ => *elem = self.integrated_field_x_envelop(*elem),
        });
        -time_part * space_part
    }
}
