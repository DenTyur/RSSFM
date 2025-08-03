use crate::config::F;
use ndarray::{Array1, Array2};
use plotters::prelude::*;

pub fn plot_heatmap_logscale(
    func: &Array2<F>,
    x: &Array1<F>,
    y: &Array1<F>,
    color_limits: (F, F),
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Проверка размеров массивов
    assert_eq!(
        func.dim(),
        (x.len(), y.len()),
        "Размеры массивов не совпадают"
    );

    // Создание области для рисования
    // let maxaxlen = x.len().max(y.len()) as f32;
    // let root = BitMapBackend::new(
    //     output_path,
    //     (810, 800),
    // (
    // (810. * x.len() as f32 / maxaxlen).round() as u32,
    // (800. * y.len() as f32 / maxaxlen).round() as u32,
    // ),
    // )
    // .into_drawing_area();
    // root.fill(&WHITE)?;

    // Определение пределов осей
    let x_min = x.iter().cloned().fold(F::INFINITY, F::min);
    let x_max = x.iter().cloned().fold(F::NEG_INFINITY, F::max);
    let y_min = y.iter().cloned().fold(F::INFINITY, F::min);
    let y_max = y.iter().cloned().fold(F::NEG_INFINITY, F::max);

    //=========================================================
    // Вычисляем соотношение сторон
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let aspect_ratio = x_range / y_range;

    // Базовые размеры (можно настроить по вашему вкусу)
    let base_width = 800.0;
    let base_height = 800.0;

    // Вычисляем размеры изображения с учетом соотношения сторон
    let (width, height) = if aspect_ratio > 1.0 {
        (base_width, base_width / aspect_ratio)
    } else {
        (base_height * aspect_ratio, base_height)
    };

    // Создание области для рисования с правильными размерами
    let root = BitMapBackend::new(output_path, (width.round() as u32, height.round() as u32))
        .into_drawing_area();
    root.fill(&WHITE)?;
    //=========================================================
    // Создание графика с основным area и area для colorbar
    let (main_area, colorbar_area) = root.split_horizontally((width * 0.95).round() as u32);
    // let (main_area, colorbar_area) = root.split_horizontally(790);

    // Основной график
    let colormin = color_limits.0;
    let colormax = color_limits.1;
    let title = format!("colorbar_limits = log {:e}, {:e}", colormin, colormax);
    let mut chart = ChartBuilder::on(&main_area)
        .caption(&title, ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // Преобразование пределов в логарифмический масштаб
    let log_min = color_limits.0.max(1e-20).ln();
    let log_max = color_limits.1.max(1e-20).ln();
    let log_range = log_min..log_max;

    // Создание цветовой палитры
    let palette = colorgrad::turbo();

    // Рисование heatmap
    for i in 0..x.len() - 1 {
        for j in 0..y.len() - 1 {
            let value = func[[i, j]];
            if value == 0.0 {
                //Float==0.0 ой как плохо, но нормально)
                let rgb = RGBColor(0, 0, 0);
                let rect = Rectangle::new([(x[i], y[j]), (x[i + 1], y[j + 1])], rgb.filled());
                chart.draw_series(std::iter::once(rect))?;
            }
            if value < 0.0 {
                continue;
            }

            let log_val = value.ln();
            let normalized = ((log_val - log_min) / (log_max - log_min)).clamp(0.0, 1.0);
            let color = palette.at(normalized as f64).to_rgba8();
            let rgb = RGBColor(color[0], color[1], color[2]);

            let rect = Rectangle::new([(x[i], y[j]), (x[i + 1], y[j + 1])], rgb.filled());
            chart.draw_series(std::iter::once(rect))?;
        }
    }

    // Рисование colorbar
    // draw_colorbar(colorbar_area, log_range, &palette, "log(Value)")?;

    Ok(())
}

// fn draw_colorbar<DB: DrawingBackend>(
//     area: DrawingArea<DB, Shift>,
//     range: std::ops::Range<F>,
//     palette: &colorgrad::Gradient,
//     label: &str,
// ) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
// where
//     DB::ErrorType: 'static,
// {
//     let mut chart = ChartBuilder::on(&area)
//         .margin(5)
//         .x_label_area_size(10)
//         .build_cartesian_2d(0..1, range.clone())?;
//
//     chart.configure_mesh().x_desc(label).draw()?;
//
//     let step = (range.end - range.start) / 100.0;
//     for y in (0..100).map(|i| range.start + i as F * step) {
//         let normalized = ((y - range.start) / (range.end - range.start)).clamp(0.0, 1.0);
//         let color = palette.at(normalized as f64).to_rgba8();
//         let rgb = RGBColor(color[0], color[1], color[2]);
//
//         let rect = Rectangle::new([(0, y), (1, y + step)], rgb.filled());
//         chart.draw_series(std::iter::once(rect))?;
//     }
//
//     Ok(())
// }
//================================================
// use plotters::prelude::*;
// use plotters::style::{RGBColor, ShapeStyle};
//
// fn draw_colorbar<DB: DrawingBackend>(
//     area: DrawingArea<DB, Shift>,
//     range: std::ops::Range<F>,
//     palette: &colorgrad::Gradient,
//     label: &str,
// ) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>>
// where
//     DB::ErrorType: 'static,
// {
//     // Создаем основную сетку для colorbar
//     let mut chart = ChartBuilder::on(&area)
//         .margin(5)
//         .x_label_area_size(10)
//         .build_cartesian_2d(0..1, range.clone())?;
//
//     // Рассчитываем основные деления
//     let major_ticks = calculate_ticks(range.start, range.end);
//
//     // Настраиваем стиль для осей
//     let axis_style = ShapeStyle {
//         color: BLACK.to_rgba(),
//         filled: false,
//         stroke_width: 1,
//     };
//
//     // Настраиваем сетку
//     chart
//         .configure_mesh()
//         .x_desc(label)
//         .disable_x_mesh()
//         .disable_x_axis()
//         .y_desc("")
//         .y_label_style(("sans-serif", 15))
//         .y_label_formatter(&|v| format!("{:.1e}", v.exp()))
//         .axis_desc_style(("sans-serif", 15))
//         .bold_line_style(WHITE.mix(0.3))
//         .light_line_style(WHITE.mix(0.1))
//         .axis_style(axis_style)
//         .y_labels(5)
//         .y_label_offset(35)
//         .draw()?;
//
//     // Вручную добавляем деления (альтернатива y_tick_values)
//     for &tick in &major_ticks {
//         chart.draw_series(std::iter::once(PathElement::new(
//             vec![(0, tick), (1, tick)],
//             ShapeStyle {
//                 color: BLACK.to_rgba(),
//                 filled: false,
//                 stroke_width: 1,
//             },
//         )))?;
//     }
//
//     // Рисуем цветовую шкалу
//     let step = (range.end - range.start) / 100.0;
//     for y in (0..100).map(|i| range.start + i as F * step) {
//         let normalized = ((y - range.start) / (range.end - range.start)).clamp(0.0, 1.0);
//         let color = palette.at(normalized as f64).to_rgba8();
//         let rgb = RGBColor(color[0], color[1], color[2]);
//
//         let rect = Rectangle::new([(0, y), (1, y + step)], rgb.filled());
//         chart.draw_series(std::iter::once(rect))?;
//     }
//
//     Ok(())
// }
//
// /// Рассчитывает позиции основных делений для логарифмической шкалы
// fn calculate_ticks(min: F, max: F) -> Vec<F> {
//     let min_exp = min.log10().floor() as i32;
//     let max_exp = max.log10().ceil() as i32;
//
//     let mut major_ticks = Vec::new();
//
//     for exp in min_exp..=max_exp {
//         let base = 10F.powi(exp);
//         major_ticks.push(base.ln());
//     }
//
//     major_ticks
// }
