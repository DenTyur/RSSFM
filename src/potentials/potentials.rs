use crate::config::F;

/// #Аналитические атомные потенциалы

//============================ 1D ========================================
/// ## 1D
/// Сглаженный кулон
pub fn soft_coulomb(x: [F; 1], a: F) -> F {
    -1.0 / (x[0] * x[0] + a * a).sqrt()
}

/// Soft Coulomb 1d E0 = 6.7534 eV, dE = 0.028 eV
pub fn soft_coulomb_e0_6_7534_eV(x: [F; 1]) -> F {
    let a: F = 3.15;
    -1.0 / (x[0] * x[0] + a * a).sqrt()
}

/// Потенциал отрицательного иона брома. Внутренний электрон. Одномерный.
pub fn br_1e1d_inner(x: [F; 1]) -> F {
    // параметры:
    let a: F = 1.66;
    // реализация:
    -1.0 / (x[0] * x[0] + a * a).sqrt()
}

/// Потенциал отрицательного иона брома. Внешний электрон. Одномерный.
pub fn br_1e1d_external(x: [F; 1]) -> F {
    // параметры:
    let c1: F = 0.48;
    let c2: F = 0.76;
    // реализация:
    -c1 * (-x[0].powi(2) / c2.powi(2)).exp()
}

//============================ 2D ========================================

/// ## 2D
/// Сглаженный кулон
pub fn soft_coulomb_2d(x: [F; 2], a: F) -> F {
    -1.0 / (x[0] * x[0] + x[1] * x[1] + a * a).sqrt()
}

/// Отрицательный ион брома. Два одномерных электрона с взаимодействием
pub fn br_2e1d(x: [F; 2]) -> F {
    let a: F = 1.66;
    let b: F = 2.6;
    -1.0 / (x[0] * x[0] + a * a).sqrt() - 1.0 / (x[1] * x[1] + a * a).sqrt()
        + 1.0 / ((x[1] - x[0]).powi(2) + b * b)
}

/// Потенциал отрицательного иона брома. Внешний электрон. Двумерный.
pub fn br_1e2d_external(x: [F; 2]) -> F {
    // параметры:
    let c1: F = 0.48;
    let c2: F = 2.0;
    let rcut: F = 500.0;
    // реализация:
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    let pot = |r: F| -c1 * (-(r / c2).powi(2)).exp();
    if r < rcut {
        pot(r)
    } else {
        pot(rcut)
    }
}

/// Потенциал отрицательного иона брома. Внутренний электрон. Двумерный.
pub fn br_1e2d_inner(x: [F; 2]) -> F {
    // параметры:
    let c: F = 1.0;
    let rcut: F = 150.0;
    // реализация:
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    let pot = |r: F| -1.0 / (r * r + c * c).sqrt();
    if r < rcut {
        pot(r)
    } else {
        pot(rcut)
    }
}

//============================ 4D ========================================

/// ## 4D
pub fn br_2e2d(x: [F; 4]) -> F {
    // параметры:
    let a: F = 1.0;
    let b: F = 2.2;
    // реализация:
    let r1_squared: F = x[0].powi(2) + x[1].powi(2);
    let r2_squared: F = x[2].powi(2) + x[3].powi(2);
    let delta_rx: F = x[2] - x[0];
    let delta_ry: F = x[3] - x[1];

    -1.0 / (r1_squared + a * a).sqrt() - 1.0 / (r2_squared + a * a).sqrt()
        + 1.0 / (delta_rx * delta_rx + delta_ry * delta_ry + b * b).sqrt()
}
