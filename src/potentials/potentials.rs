use crate::config::F;

/// #Аналитические атомные потенциалы

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
