use crate::config::{C, F, I};

/// # Поглощающие комплексные потенциалы

pub fn absorbing_potential(x: [F; 2], r0: F, alpha: F) -> C {
    // параметры потенциала:
    // let r0: F = 50.0;
    // let alpha: F = 0.02;
    // реализация:
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    if r > r0 {
        -I * (r - r0) * alpha * (1.0 - (-0.5 * (r - r0)).exp())
    } else {
        C::new(0.0, 0.0)
    }
}
