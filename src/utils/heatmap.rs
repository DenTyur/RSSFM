use crate::config::F;
use colorous::{Gradient, TURBO};
use ndarray::{prelude::*, stack, Zip};
use plotters::{coord::types::RangedCoordf32, prelude::*};
use std::error::Error; // Neg - оператор арифметического отрицания(минус)

// ????????????????????????????????????????????????????????????????
fn filled_style<C: Into<RGBAColor>>(color: C) -> ShapeStyle {
    ShapeStyle {
        color: color.into(),
        filled: true,
        stroke_width: 0,
    }
}

struct Colorbar {
    min: F,
    max: F,
    gradient: Gradient,
}

impl Colorbar {
    fn color(&self, value: F) -> RGBColor {
        let &Self {
            min,
            max,
            gradient: colormap,
        } = self;
        let value = value.max(min).min(max);
        let (r, g, b) = colormap
            .eval_continuous((value - min) as f64 / (max - min) as f64)
            .as_tuple();
        RGBColor(r, g, b)
    }

    fn color_log(&self, value: F) -> RGBColor {
        let &Self {
            min,
            max,
            gradient: colormap,
        } = self;
        let value = value.max(min).min(max); // ГЕНИАЛЬНО!!!
        let t = (value.log10() - min.log10()) / (max.log10() - min.log10());
        println!("{}", t);
        let (r, g, b) = colormap.eval_continuous(t as f64).as_tuple();
        RGBColor(r, g, b)
    }

    fn draw<DB: DrawingBackend>(&self, text_color: RGBColor, mut chart_builder: ChartBuilder<DB>) {
        let &Self { min, max, .. } = self;
        let step = (max - min) / 256.0;
        let mut chart_context = chart_builder
            .margin_top(10)
            .x_label_area_size(30)
            .y_label_area_size(0)
            .right_y_label_area_size(45)
            .build_cartesian_2d(0.0..1.0, min..max)
            .unwrap()
            .set_secondary_coord(0.0..1.0, min..max);

        chart_context
            .configure_mesh()
            .set_all_tick_mark_size(0)
            .disable_x_axis()
            .disable_y_axis()
            .disable_x_mesh()
            .disable_y_mesh()
            .axis_style(&text_color)
            .label_style("sans-serif".into_font().color(&text_color))
            .draw()
            .unwrap();

        chart_context
            .configure_secondary_axes()
            // .y_label_formatter(&|x| format!("{:e}", x))
            .axis_style(&text_color)
            .label_style("sans-serif".into_font().color(&text_color))
            .draw()
            .unwrap();

        let plotting_area = chart_context.plotting_area();
        let values = Array1::range(min, max + step, step);
        for value in values {
            let color = self.color(value);
            let rectangle = Rectangle::new(
                [(0.0, (value - step / 2.0)), (1.0, (value + step / 2.0))],
                filled_style(color),
            );
            plotting_area.draw(&rectangle).unwrap();
        }
    }
}

fn meshgrid<A: Clone>(x: &Array1<A>, y: &Array1<A>) -> (Array2<A>, Array2<A>) {
    (
        stack(Axis(0), &vec![x.view(); y.len()]).unwrap(),
        stack(Axis(1), &vec![y.view(); x.len()]).unwrap(),
    )
}

fn heatmap<'a, 'b, 'c, DB: DrawingBackend>(
    function: &Array2<F>,
    x_arr: &Array1<F>,
    y_arr: &Array1<F>,
    colorbar: &Colorbar,
    text_color: RGBAColor,
    mut chart_builder: ChartBuilder<'b, 'c, DB>,
) -> Result<ChartContext<'b, DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>, Box<dyn Error>> {
    let (x_grid, y_grid) = meshgrid(&x_arr, &y_arr);

    let mut chart_context = chart_builder
        .margin_top(10)
        .x_label_area_size(25)
        .right_y_label_area_size(0)
        .y_label_area_size(25)
        .build_cartesian_2d(
            x_arr[0]..x_arr[x_arr.len() - 1],
            y_arr[0]..y_arr[y_arr.len() - 1],
        )
        .unwrap();

    chart_context
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .set_all_tick_mark_size(5)
        .axis_style(&text_color)
        .label_style("sans-serif".into_font().color(&text_color))
        .draw()
        .unwrap();

    let plotting_area = chart_context.plotting_area();
    let step = x_arr[1] - x_arr[0];

    Zip::from(&x_grid)
        .and(&y_grid)
        .and(function)
        .for_each(|&x, &y, &f| {
            let rectangle = Rectangle::new(
                [
                    (x - step / 2.0, y - step / 2.0),
                    (x + step / 2.0, y + step / 2.0),
                ],
                filled_style(colorbar.color(f)),
            );
            plotting_area.draw(&rectangle).unwrap();
        });

    Ok(chart_context)
}

pub fn plot_heatmap(
    x_arr: &Array1<F>,
    y_arr: &Array1<F>,
    func: &Array2<F>,
    size_x: u32,
    size_y: u32,
    size_colorbar: u32,
    colorbar_min: F,
    colorbar_max: F,
    save_path: &str,
) {
    let drawing_area =
        BitMapBackend::new(save_path, (size_x + size_colorbar, size_y)).into_drawing_area();
    drawing_area.fill(&WHITE).unwrap();

    let (left, right) = drawing_area.split_horizontally(size_x);
    let colorbar = Colorbar {
        min: colorbar_min,
        max: colorbar_max,
        gradient: TURBO,
    };
    colorbar.draw(BLACK.into(), ChartBuilder::on(&right));

    let mut chart_builder = ChartBuilder::on(&left);
    chart_builder.margin(10);

    heatmap(func, x_arr, y_arr, &colorbar, BLACK.into(), chart_builder)
        .expect("failure while drawing heatmap");

    drawing_area.present().expect("failure while writing file");
}

#[test]
fn test_heatmap() {
    let (size_x, size_y, size_colorbar) = (500, 500, 60);
    let (dx, dy) = (1e-2, 1e-2);
    let x_arr = Array1::range(-5.0, 5.0 + dx, dx);
    let y_arr = Array1::range(-5.0, 5.0 + dy, dy);
    let func = function(&x_arr, &y_arr);
    let (colorbar_min, colorbar_max) = (1e-3, 1e-1);

    plot_heatmap(
        &x_arr,
        &y_arr,
        &func,
        size_x,
        size_y,
        size_colorbar,
        colorbar_min,
        colorbar_max,
        "wave.png",
    )
}

fn function(x_arr: &Array1<F>, y_arr: &Array1<F>) -> Array2<F> {
    let mut function: Array2<F> = Array::zeros((x_arr.len(), y_arr.len()));
    function
        .axis_iter_mut(Axis(0))
        .zip(x_arr.iter())
        .for_each(|(mut function_row, x_i)| {
            function_row
                .iter_mut()
                .zip(y_arr.iter())
                .for_each(|(f, y_j)| {
                    *f = (-1.0 * (x_i.powi(2) + y_j.powi(2))).exp();
                })
        });
    function
}
