use crate::config::{F, PI};
use crate::traits::field::Field;
use std::marker::Copy;

/// 1e3d unipolar
/// Струкрура задающая униполярное поле вдоль оси z в 3D пространстве.
/// Один трехмерный электрон.
#[derive(Debug, Clone, Copy)]
pub struct UnipolarPulse1e3d {
    pub amplitude: F,
    pub omega: F,
    pub x_envelop: F,
}

impl UnipolarPulse1e3d {
    pub fn new(amplitude: F, omega: F, x_envelop: F) -> Self {
        Self {
            amplitude,
            omega,
            x_envelop,
        }
    }

    pub fn electric_field_time_dependence(&self, t: F) -> [F; 3] {
        let mut electric_field: F = 0.;
        if PI / self.omega - t > 0. {
            electric_field = -self.amplitude * F::sin(self.omega * t).powi(2);
        }
        [0.0, 0.0, electric_field]
    }

    pub fn field_x_envelop(&self, x: F) -> F {
        // Пространственная огибающая электрического поля вдоль каждой из осей.
        F::cos(PI / 2. * x / self.x_envelop).powi(2)
    }

    pub fn integrated_field_x_envelop(&self, x: F) -> F {
        0.5 * x + 0.25 * self.x_envelop * 2. / PI * F::sin(PI * x / self.x_envelop)
    }

    pub fn electric_field(&self, t: F, x: [F; 3]) -> [F; 3] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [
            self.electric_field_time_dependence(t)[0] * self.field_x_envelop(x[0]),
            self.electric_field_time_dependence(t)[1] * self.field_x_envelop(x[1]),
            self.electric_field_time_dependence(t)[2] * self.field_x_envelop(x[2]),
        ]
    }
}

impl Field<3> for UnipolarPulse1e3d {
    fn vector_potential(&self, t: F) -> [F; 3] {
        println!("Векторный потенциал не реализован! Заглушка.");
        let vec_pot: F = 0.0;
        [vec_pot, vec_pot, vec_pot]
    }

    fn scalar_potential(&self, x: [F; 3], t: F) -> F {
        let x0_point = x[0];
        let time_part0: F = self.electric_field_time_dependence(t)[0];
        let space_part0: F = match x0_point {
            x if x <= -self.x_envelop => self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => self.integrated_field_x_envelop(self.x_envelop),
            _ => self.integrated_field_x_envelop(x0_point),
        };

        let x1_point = x[1];
        let time_part1: F = self.electric_field_time_dependence(t)[1];
        let space_part1: F = match x1_point {
            x if x <= -self.x_envelop => self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => self.integrated_field_x_envelop(self.x_envelop),
            _ => self.integrated_field_x_envelop(x1_point),
        };

        let x2_point = x[2];
        let time_part2: F = self.electric_field_time_dependence(t)[2];
        let space_part2: F = match x2_point {
            x if x <= -self.x_envelop => self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => self.integrated_field_x_envelop(self.x_envelop),
            _ => self.integrated_field_x_envelop(x2_point),
        };
        -time_part0 * space_part0 - time_part1 * space_part1 - time_part2 * space_part2
    }
    fn a(&self, t: F) -> [F; 3] {
        println!("Векторный потенциал не реализован! Заглушка.");
        let a: F = 0.0;
        [a, a, a]
    }
    fn b(&self, t: F) -> F {
        println!("Векторный потенциал не реализован! Заглушка.");
        let b: F = 0.0;
        b
    }
}
