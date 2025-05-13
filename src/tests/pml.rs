use super::super::pml::PML;
use crate::config;
use crate::parameters;
use config::{Float, PI};
use parameters::Xspace;
use plotters::prelude::*;

#[test]
fn pml_gauss() {}

#[test]
fn plot_pml() {
    let x_dir_path = "src/arrays_saved";
    let x = Xspace::load(x_dir_path, 2);

    let width = 80.0 * x.dx[0];
    let n: i32 = 3;
    let sigma0 = ((n + 1) as Float) * 2.0 * PI / x.dx[0] / (2.0 * width);
    let omega: Float = 1.0;
    let pml = PML::new(sigma0, omega, width, n, &x);

    // Создаём график
    let root = BitMapBackend::new("src/tests/out/pml/pml.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Полиномиальный профиль PML (n=3)", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            x.grid[0][0]..x.grid[0][x.n[0] - 1],
            -0.1 * sigma0..sigma0 * 1.1,
        )
        .unwrap();

    // Оси и сетка
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("σₓ(x)")
        .draw()
        .unwrap();

    // Рисуем профиль
    chart
        .draw_series(LineSeries::new(
            x.grid[0]
                .iter()
                .zip(pml.sigma[0].iter())
                .map(|(&x, &s)| (x, s)),
            &RED,
        ))
        .unwrap();
}
