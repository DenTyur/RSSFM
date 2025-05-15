use super::gauge::{LenthGauge2D, VelocityGauge2D};
use crate::config::{C, F, I, PI};
use crate::traits::{volkov::VolkovGauge, wave_function::ValueAndSpaceDerivatives};

//=================================================================================
//             фаза и множители при производных в разных калибровках
//=================================================================================
/// Фаза и множитель при производной в калибровке скорости
impl<'a> VolkovGauge for VelocityGauge2D<'a> {
    fn compute_phase(&self, x: [F; 2], p: [F; 2], t: F) -> F {
        let p_sq = p[0].powi(2) + p[1].powi(2);
        let a = self.field.a(t);
        -0.5 * t * p_sq + (p[0] * x[0] + p[1] * x[1]) - (p[0] * a[0] + p[1] * a[1])
    }

    fn deriv_factor(&self, p: [F; 2], _t: F) -> [C; 2] {
        [I * p[0], I * p[1]]
    }
}

/// Фаза и множитель при производной в калибровке длины
impl<'a> VolkovGauge for LenthGauge2D<'a> {
    fn compute_phase(&self, x: [F; 2], p: [F; 2], t: F) -> F {
        let p_sq = p[0].powi(2) + p[1].powi(2);
        let vec_pot = self.field.vec_pot(t);
        let a = self.field.a(t);
        let b = self.field.b(t);
        -0.5 * t * p_sq + (p[0] * x[0] + p[1] * x[1]) - (p[0] * vec_pot[0] + p[1] * vec_pot[1])
            + (p[0] * a[0] + p[1] * a[1])
            - 0.5 * b
    }

    fn deriv_factor(&self, p: [F; 2], t: F) -> [C; 2] {
        let vec_pot = self.field.vec_pot(t);
        [I * (p[0] - vec_pot[0]), I * (p[1] - vec_pot[1])]
    }
}

//=================================================================================
//                            Волковская функция
//=================================================================================
/// Структура для Волковских функций
pub struct Volkov2D<'a, G: VolkovGauge> {
    gauge: &'a G,
    pub p: [F; 2],
    pub t: F,
}

impl<'a, G: VolkovGauge> Volkov2D<'a, G> {
    pub fn new(gauge: &'a G, p: [F; 2], t: F) -> Self {
        Self { gauge, p, t }
    }
}

/// Значение Волковской функции в точке и производная в точке
impl<'a, G: VolkovGauge> ValueAndSpaceDerivatives<2> for Volkov2D<'a, G> {
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
