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

/// 1D асимметричный
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

/// 2D асимметричный
pub fn absorbing_potential_2d_asim(
    point: [F; 2],       // Точка в 2D пространстве [x, y]
    bounds: [[F; 2]; 2], // Границы области [[x_min, x_max], [y_min, y_max]]
    alpha: F,            // Параметр поглощения
) -> C {
    let x = point[0];
    let y = point[1];

    let x_min = bounds[0][0];
    let x_max = bounds[0][1];
    let y_min = bounds[1][0];
    let y_max = bounds[1][1];

    let mut potential = C::new(0.0, 0.0);

    // Левая граница по x (x < x_min)
    if x < x_min {
        let dist = x_min - x;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Правая граница по x (x > x_max)
    else if x > x_max {
        let dist = x - x_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по y (y < y_min)
    if y < y_min {
        let dist = y_min - y;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по y (y > y_max)
    else if y > y_max {
        let dist = y - y_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    potential
}

/// 3D асимметричный
pub fn absorbing_potential_3d_asim(
    point: [F; 3],       // Точка в 3D пространстве [x, y, z]
    bounds: [[F; 2]; 3], // Границы области [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    alpha: F,            // Параметр поглощения
) -> C {
    let x = point[0];
    let y = point[1];
    let z = point[2];

    let x_min = bounds[0][0];
    let x_max = bounds[0][1];
    let y_min = bounds[1][0];
    let y_max = bounds[1][1];
    let z_min = bounds[2][0];
    let z_max = bounds[2][1];

    let mut potential = C::new(0.0, 0.0);

    // Левая граница по x (x < x_min)
    if x < x_min {
        let dist = x_min - x;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Правая граница по x (x > x_max)
    else if x > x_max {
        let dist = x - x_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по y (y < y_min)
    if y < y_min {
        let dist = y_min - y;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по y (y > y_max)
    else if y > y_max {
        let dist = y - y_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по z (z < z_min)
    if z < z_min {
        let dist = z_min - z;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по z (z > z_max)
    else if z > z_max {
        let dist = z - z_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    potential
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

/// 4D асимметричный
pub fn absorbing_potential_4d_asim(point: [F; 4], bounds: [[F; 2]; 4], alpha: F) -> C {
    let x = point[0];
    let y = point[1];
    let z = point[2];
    let h = point[3];

    let x_min = bounds[0][0];
    let x_max = bounds[0][1];
    let y_min = bounds[1][0];
    let y_max = bounds[1][1];
    let z_min = bounds[2][0];
    let z_max = bounds[2][1];
    let h_min = bounds[3][0];
    let h_max = bounds[3][1];

    let mut potential = C::new(0.0, 0.0);

    // Левая граница по x (x < x_min)
    if x < x_min {
        let dist = x_min - x;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Правая граница по x (x > x_max)
    else if x > x_max {
        let dist = x - x_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по y (y < y_min)
    if y < y_min {
        let dist = y_min - y;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по y (y > y_max)
    else if y > y_max {
        let dist = y - y_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по z (z < z_min)
    if z < z_min {
        let dist = z_min - z;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по z (z > z_max)
    else if z > z_max {
        let dist = z - z_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    // Нижняя граница по h (h < h_min)
    if h < h_min {
        let dist = h_min - h;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }
    // Верхняя граница по h (h > h_max)
    else if h > h_max {
        let dist = h - h_max;
        potential -= I * dist * alpha * (1.0 - (-0.5 * dist).exp());
    }

    potential
}
