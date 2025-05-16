use crate::config::F;

/// #Аналитические атомные потенциалы

//============================ 2D ========================================

/// ## 2D
/// Потенциал отрицательного иона брома. Внешний электрон. Двумерный.
pub fn br_1e2d_external(x: [F; 2]) -> F {
    // параметры:
    let c1: F = 0.48;
    let c2: F = 2.0;
    let rcut: F = 20.0;
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
