use crate::config::F;
use crate::traits::tsurff::Tsurff;
use ndarray::prelude::*;
use plotters::prelude::*;
use rayon::prelude::*;

pub fn plot_log(x_values: Array1<F>, psi_norm_sq: Array1<F>, xlabel: &str, file_path: &str) {
    // Создаем область для рисования
    let root = BitMapBackend::new(file_path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Находим минимальное и максимальное значения для осей
    let x_min = x_values[[0]];
    let x_max = x_values[[x_values.len() - 1]];

    // Для логарифмической оси y находим диапазон значений (исключаем нули и отрицательные)
    let y_min = psi_norm_sq
        .iter()
        .filter(|&&y| y > 0.0)
        .fold(F::INFINITY, |a, &b| a.min(b))
        .log10();
    let y_max = psi_norm_sq
        .iter()
        .filter(|&&y| y > 0.0)
        .fold(F::NEG_INFINITY, |a, &b| a.max(b))
        .log10();

    // Создаем график с линейной осью x и логарифмической осью y
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    // Настраиваем ось y как логарифмическую
    chart
        .configure_mesh()
        .x_desc("xlabel")
        .y_label_formatter(&|y| format!("10^{:.1}", y))
        .draw()
        .unwrap();

    // Рисуем график
    chart
        .draw_series(LineSeries::new(
            x_values
                .iter()
                .zip(psi_norm_sq.iter())
                .map(|(&x, y)| (x, y.log10())),
            BLUE.stroke_width(1),
        ))
        .unwrap();
}
