use crate::config::{C, F, I};

/// #Аналитические атомные потенциалы

/// Потенциал отрицательного иона брома. Внешний электрон. Двумерный.
pub fn br_1e2d_external(x: [F; 2]) -> F {
    let c1: F = 0.48;
    let c2: F = 2.0;
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    // let pot = |r: F| -c1 * (-(x / c2).powi(2) - (y / c2).powi(2)).exp();
    let pot = |r: F| -c1 * (-(r / c2).powi(2)).exp();
    if r < 20.0 {
        pot(r)
    } else {
        pot(20.0)
    }
}

/// #Комплексные поглощающие потенциалы
pub fn absorbing_potential(x: [F; 2]) -> C {
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    let r0: F = 30.0;
    let alpha: F = 0.02;
    if r > r0 {
        -I * (r - r0) * alpha * (1.0 - (-0.5 * (r - r0)).exp())
    } else {
        C::new(0.0, 0.0)
    }
}

pub fn absorbing_potential_linear(x: [F; 2]) -> C {
    let r: F = (x[0].powi(2) + x[1].powi(2)).sqrt();
    let r0: F = 50.0;
    let alpha: F = 0.02;
    if r > r0 {
        -I * (r - r0).abs() * alpha
    } else {
        C::new(0.0, 0.0)
    }
}

#[test]
fn cap_linear() {
    use crate::dim2::heatmap;
    use crate::dim2::space::Xspace2D;
    use crate::traits::space::Space;
    use ndarray::prelude::*;
    let x_dir_path = "src/arrays_saved";
    let x = Xspace2D::load_from_npy(x_dir_path);

    let mut cap_arr: Array<F, Ix2> = Array::zeros((x.n[0], x.n[1]));
    let mut colorbar_max: F = 0.0;
    let mut colorbar_min: F = 1e5;
    cap_arr
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .for_each(|(mut cap_row, x_point)| {
            cap_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(cap_elem, y_point)| {
                    *cap_elem = absorbing_potential_linear([*x_point, *y_point]).norm();
                    if *cap_elem > colorbar_max {
                        colorbar_max = *cap_elem;
                    }
                    if *cap_elem < colorbar_min {
                        colorbar_min = *cap_elem;
                    }
                });
        });

    // Создаём график
    let (size_x, size_y, size_colorbar) = (500, 500, 60);

    heatmap::plot_heatmap(
        &x.grid[0],
        &x.grid[1],
        &cap_arr,
        size_x,
        size_y,
        size_colorbar,
        colorbar_min,
        colorbar_max,
        "cap.png",
    )
}

#[test]
fn cap() {
    use crate::dim2::heatmap;
    use crate::dim2::space::Xspace2D;
    use crate::traits::space::Space;
    use ndarray::prelude::*;

    let x_dir_path = "src/arrays_saved";
    let x = Xspace2D::load_from_npy(x_dir_path);

    let mut cap_arr: Array<F, Ix2> = Array::zeros((x.n[0], x.n[1]));
    let mut colorbar_max: F = 0.0;
    let mut colorbar_min: F = 1e5;
    cap_arr
        .axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .for_each(|(mut cap_row, x_point)| {
            cap_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(cap_elem, y_point)| {
                    *cap_elem = absorbing_potential([*x_point, *y_point]).norm();
                    if *cap_elem > colorbar_max {
                        colorbar_max = *cap_elem;
                    }
                    if *cap_elem < colorbar_min {
                        colorbar_min = *cap_elem;
                    }
                });
        });

    // Создаём график
    let (size_x, size_y, size_colorbar) = (500, 500, 60);

    heatmap::plot_heatmap(
        &x.grid[0],
        &x.grid[1],
        &cap_arr,
        size_x,
        size_y,
        size_colorbar,
        colorbar_min,
        colorbar_max,
        "cap.png",
    )
}
