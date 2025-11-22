use crate::config::{F, PI};
use crate::traits::field::Field;

/// Постоянное поле
pub struct ConstantField1D {
    pub amplitude: F,
}

impl ConstantField1D {
    pub fn new(amplitude: F) -> Self {
        Self { amplitude }
    }
    pub fn electric_field_time_dependence(&self, t: F) -> F {
        // Возвращает электрическое поле в момент времени t вдоль
        // каждой из пространственных осей x0, x1 и т.д.: массив размерности dim.
        // Каждый элемент этого массива содержит электрическое поле
        // в момент времени t вдоль соответствующей оси.
        // Например, E0 = electric_fielf(2.)[0] - электрическое
        // поле в момент времени t=2 вдоль оси x0.

        self.amplitude
    }

    pub fn field_x_envelop(&self, x: F) -> F {
        // Пространственная огибающая электрического поля вдоль каждой из осей.
        1.0
    }

    pub fn integrated_field_x_envelop(&self, x: F) -> F {
        x
    }

    pub fn electric_field(&self, t: F, x: F) -> [F; 1] {
        // Электрическое поле вдоль каждой пространственной оси в момент времени t.
        [self.electric_field_time_dependence(t) * self.field_x_envelop(x)]
    }
}

impl Field<1> for ConstantField1D {
    fn vector_potential(&self, t: F) -> [F; 1] {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let vec_pot: F = 0.0;
        [vec_pot]
    }

    fn scalar_potential(&self, x: [F; 1], t: F) -> F {
        let x_point = x[0];
        let time_part: F = self.electric_field_time_dependence(t);
        let space_part: F = self.integrated_field_x_envelop(x_point);
        -time_part * space_part
    }
    fn a(&self, t: F) -> [F; 1] {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let a: F = 0.0;
        [a]
    }
    fn b(&self, t: F) -> F {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let b: F = 0.0;
        b
    }
}
/// Униполярный импульс
pub struct UnipolarPulse1D {
    pub amplitude: F,
    pub omega: F,
    pub x_envelop: F,
}

impl UnipolarPulse1D {
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
}

impl Field<1> for UnipolarPulse1D {
    fn vector_potential(&self, t: F) -> [F; 1] {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let vec_pot: F = 0.0;
        [vec_pot]
    }

    fn scalar_potential(&self, x: [F; 1], t: F) -> F {
        let x_point = x[0];
        let time_part: F = self.electric_field_time_dependence(t);
        let space_part: F = match x_point {
            x if x <= -self.x_envelop => self.integrated_field_x_envelop(-self.x_envelop),
            x if x >= self.x_envelop => self.integrated_field_x_envelop(self.x_envelop),
            _ => self.integrated_field_x_envelop(x_point),
        };
        -time_part * space_part
    }
    fn a(&self, t: F) -> [F; 1] {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let a: F = 0.0;
        [a]
    }
    fn b(&self, t: F) -> F {
        panic!("Векторный потенциал не реализован! Заглушка.");
        let b: F = 0.0;
        b
    }
}
