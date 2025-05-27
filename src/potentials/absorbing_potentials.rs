use crate::config::{C, F, I};

/// # Поглощающие комплексные потенциалы

/// 1D
pub fn absorbing_potential_1d(x: [F; 1], r0: F, alpha: F) -> C {
    let r: F = x[0].abs();
    if r > r0 {
        -I * (r - r0) * alpha * (1.0 - (-0.5 * (r - r0)).exp())
    } else {
        C::new(0.0, 0.0)
    }
}

/// асимметричный
pub fn absorbing_potential_1d_asim(x: [F; 1], r0: [F; 2], alpha: F) -> C {
    let x_val = x[0];
    if x_val < r0[0] {
        // Левая поглощающая область
        -I * (r0[0] - x_val) * alpha * (1.0 - (-0.5 * (r0[0] - x_val)).exp())
    } else if x_val > r0[1] {
        // Правая поглощающая область
        -I * (x_val - r0[1]) * alpha * (1.0 - (-0.5 * (x_val - r0[1])).exp())
    } else {
        // Центральная (непоглощающая) область
        C::new(0.0, 0.0)
    }
}

/// 2D
pub fn absorbing_potential_2d(x: [F; 2], r0: F, alpha: F) -> C {
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

/// 4D
pub fn absorbing_potential_4d(x: [F; 4], r0: F, alpha: F) -> C {
    // параметры потенциала:
    // let r0: F = 50.0;
    // let alpha: F = 0.02;
    // реализация:
    let r: F = (x[0].powi(2) + x[1].powi(2) + x[2].powi(2) + x[3].powi(2)).sqrt();
    if r > r0 {
        -I * (r - r0) * alpha * (1.0 - (-0.5 * (r - r0)).exp())
    } else {
        C::new(0.0, 0.0)
    }
}
