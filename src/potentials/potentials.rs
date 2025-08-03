use crate::config::F;

/// #Аналитические атомные потенциалы

//============================ 1D ========================================
/// ## 1D
/// Сглаженный кулон
pub fn soft_coulomb(x: [F; 1], a: F) -> F {
    -1.0 / (x[0] * x[0] + a * a).sqrt()
}

// Сглаженный кулон 1D
// z -- заряд остова с учетом знака (-1 для атома)
// a -- параметр сглаживания
pub fn soft_coulomb_1d(x: [F; 1], z: F, a: F) -> F {
    z / (x[0] * x[0] + a * a).sqrt()
}

/// Soft Coulomb 1d E0 = 6.7534 eV, dE = 0.028 eV
pub fn soft_coulomb_e0_6_7534_eV(x: [F; 1]) -> F {
    let a: F = 3.15;
    -1.0 / (x[0] * x[0] + a * a).sqrt()
}

/// Soft Coulomb 1d E0 = 9.03 eV, dE = 0.028 eV
pub fn soft_coulomb_e0_9_03_eV(x: [F; 1]) -> F {
    let a: F = 2.27;
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

pub fn short_e0_5_899eV(x: [F; 1]) -> F {
    // параметры:
    let c1: F = 0.69;
    let c2: F = 0.76;
    // реализация:
    -c1 * (-x[0].powi(2) / c2.powi(2)).exp()
}

/// Короткодействующий потенциал
pub fn short_c2_076(x: [F; 1], c1: F) -> F {
    // параметры:
    // let c1: F = 0.48;
    let c2: F = 0.76;
    // реализация:
    -c1 * (-x[0].powi(2) / c2.powi(2)).exp()
}

//============================ 2D ========================================

/// ## 2D
/// Гармонический осциллятор
pub fn oscillator_2d(x: [F; 2]) -> F {
    0.5 * (x[0] * x[0] + x[1] * x[1])
}
/// Сглаженный кулон двумерный
/// z -- заряд остова с учетом знака (-1 для атома)
/// a -- сглаживающий параметр
pub fn soft_coulomb_2d(x: [F; 2], z: F, a: F) -> F {
    z / (x[0] * x[0] + x[1] * x[1] + a * a).sqrt()
}

/// soft_coulomb_2e1d
pub fn soft_coulomb_2e1d(x: [F; 2], b: F) -> F {
    let a: F = 3.15;
    // let b: F = ;
    -1.0 / (x[0].powi(2) + a.powi(2)).sqrt() - 1.0 / (x[1].powi(2) + a.powi(2)).sqrt()
        + 1.0 / ((x[1] - x[0]).powi(2) + b.powi(2)).sqrt()
}

pub fn soft_coulomb_2e1d_e0_10_1159_eV(x: [F; 2]) -> F {
    let a: F = 3.15;
    let b: F = 7.43;
    -1.0 / (x[0].powi(2) + a.powi(2)).sqrt() - 1.0 / (x[1].powi(2) + a.powi(2)).sqrt()
        + 1.0 / ((x[1] - x[0]).powi(2) + b.powi(2)).sqrt()
}

// Сглаженный кулон 2e1d с взаимодействием
// z -- заряд остова с учетом знака (-2 для атома)
// a -- сглаживающий параметр взаимодействия электронов с остовом
// b -- сглаживающий параметр e-e взаимодействия
pub fn soft_coulomb_2e1d_interact(x: [F; 2], z: F, a: F, b: F) -> F {
    z / (x[0].powi(2) + a.powi(2)).sqrt()
        + z / (x[1].powi(2) + a.powi(2)).sqrt()
        + 1.0 / ((x[1] - x[0]).powi(2) + b.powi(2)).sqrt()
}

/// Отрицательный ион брома. Два одномерных электрона с взаимодействием
pub fn br_2e1d(x: [F; 2]) -> F {
    let a: F = 1.66;
    let b: F = 2.6;
    -1.0 / (x[0].powi(2) + a.powi(2)).sqrt() - 1.0 / (x[1].powi(2) + a.powi(2)).sqrt()
        + 1.0 / ((x[1] - x[0]).powi(2) + b.powi(2)).sqrt()
}

/// Отрицательный ион брома. Два одномерных электрона с взаимодействием. Центр масс
pub fn br_2e1d_com(x: [F; 2]) -> F {
    let a: F = 1.66;
    let b: F = 2.6;
    let r: F = x[0];
    let R: F = x[1];
    -1.0 / ((R + r).powi(2) + a.powi(2)).sqrt() - 1.0 / ((R - r).powi(2) + a.powi(2)).sqrt()
        + 1.0 / (r.powi(2) + b.powi(2)).sqrt()
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

//============================ 3D ========================================

/// ## 3D
/// Гармонический осциллятор
pub fn oscillator_3d(x: [F; 3]) -> F {
    0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
}

/// Трехмерный одноэлектронный гелий He3d
pub fn He_3d(x: [F; 3]) -> F {
    let z: F = 1.0;
    let a1: F = 1.375;
    let a2: F = 0.662;
    let a3: F = -1.325;
    let a4: F = 1.236;
    let a5: F = -0.231;
    let a6: F = 0.480;
    let a: F = 0.1;
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + a * a).sqrt();
    -(z + a1 * (-a2 * r).exp() + a3 * r * (-a4 * r).exp() + a5 * (-a6 * r).exp()) / r
}

//============================ 4D ========================================

/// Сглаженный кулон с взаимодействием
pub fn soft_coulomb_2e2d_interact(x: [F; 4], z: F, a: F, b: F) -> F {
    // реализация:
    let r1_squared: F = x[0].powi(2) + x[1].powi(2);
    let r2_squared: F = x[2].powi(2) + x[3].powi(2);
    let delta_rx: F = x[2] - x[0];
    let delta_ry: F = x[3] - x[1];

    z / (r1_squared + a * a).sqrt()
        + z / (r2_squared + a * a).sqrt()
        + 1.0 / (delta_rx * delta_rx + delta_ry * delta_ry + b * b).sqrt()
}

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

pub fn br_2e2d_com(x: [F; 4]) -> F {
    // r: 0,1
    // R: 2,3
    let r1: [F; 2] = [x[2] + x[0], x[3] + x[1]];
    let r2: [F; 2] = [x[2] - x[0], x[3] - x[1]];
    // параметры:
    let a: F = 1.0;
    let b: F = 2.2;
    // реализация:
    let r1_squared: F = r1[0].powi(2) + r1[1].powi(2);
    let r2_squared: F = r2[0].powi(2) + r2[1].powi(2);
    let delta_rx: F = x[0];
    let delta_ry: F = x[1];

    -1.0 / (r1_squared + a * a).sqrt() - 1.0 / (r2_squared + a * a).sqrt()
        + 1.0 / (delta_rx * delta_rx + delta_ry * delta_ry + b * b).sqrt()
}
