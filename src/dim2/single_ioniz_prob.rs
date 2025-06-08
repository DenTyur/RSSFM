use super::{
    fft_maker::FftMaker2D,
    space::{Pspace2D, Xspace2D},
    wave_function::WaveFunction2D,
};
use crate::common::tspace::Tspace;
use crate::config::{C, F, I, PI};
use crate::macros::check_path;
use crate::traits::{
    fft_maker::FftMaker,
    wave_function::{ValueAndSpaceDerivatives, WaveFunction},
};
use crate::utils::hdf5_interface;
use itertools::multizip;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use plotters::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

#[derive(Debug, Clone)]
pub struct SingleIonizProb2D<const N: usize> {
    pub r_surf: [F; N],
    pub ioniz_prob: [Vec<F>; N],
    pub x: Xspace2D,
    pub t: Array1<F>,
}

impl<const N: usize> SingleIonizProb2D<N> {
    pub fn new(r_surf: [F; N], x: Xspace2D, t: Array1<F>) -> Self {
        let ioniz_prob: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
        Self {
            r_surf: r_surf,
            ioniz_prob,
            x,
            t,
        }
    }

    pub fn save_as_hdf5(&self, path: &str) {
        let r_surf_arr = Array1::from_vec(self.r_surf.to_vec());
        hdf5_interface::write_to_hdf5(path, "x_surf", None, &r_surf_arr).unwrap();
        hdf5_interface::write_to_hdf5(path, "t", None, &self.t).unwrap();
        for i in 0..N {
            let ioniz_prob_i = Array1::from_vec(self.ioniz_prob[i].to_vec());
            let lenth = ioniz_prob_i.len();
            let last = ioniz_prob_i[lenth - 1];
            hdf5_interface::write_to_hdf5(
                path,
                format!("ioniz_prob_{i}").as_str(),
                None,
                &ioniz_prob_i,
            )
            .unwrap();
            hdf5_interface::create_str_data_attr(
                path,
                format!("ioniz_prob_{i}").as_str(),
                None,
                "last",
                format!("{last}").as_str(),
            )
            .unwrap();
        }
    }

    pub fn add(&mut self, wf: &Array2<C>) {
        for i in 0..N {
            let mut ioniz_prob_instance: F = 0.0;
            wf.axis_iter(Axis(0))
                .zip(self.x.grid[0].iter())
                .for_each(|(psi_row, x_elem)| {
                    psi_row
                        .iter()
                        .zip(self.x.grid[1].iter())
                        .for_each(|(psi_elem, y_elem)| {
                            if x_elem * x_elem + y_elem * y_elem > self.r_surf[i] {
                                ioniz_prob_instance += psi_elem.norm_sqr()
                            }
                        });
                });
            self.ioniz_prob[i].push(ioniz_prob_instance * self.x.dx[0] * self.x.dx[1]);
        }
    }

    pub fn plot(&self, t: &Array1<F>, output_path: &str) {
        // Создаём область для рисования
        // let t = tspace.grid.clone();

        let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut y_max: F = 0.0;
        for i in 0..N {
            let max = self.ioniz_prob[i]
                .iter()
                .fold(F::NEG_INFINITY, |a, &b| a.max(b));
            if max > y_max {
                y_max = max;
            }
        }

        let mut chart = ChartBuilder::on(&root)
            .caption("Вероятность ионизации", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((t[0]..t[t.len() - 1].into()), 0.0f64..y_max as f64 * 1.1)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("t [au]")
            .y_desc("Вероятность ионизации")
            .draw()
            .unwrap();

        // Создаем палитру цветов
        let colors = [
            RED,
            GREEN,
            BLUE,
            MAGENTA,
            CYAN,
            YELLOW,
            RGBColor(255, 165, 0), // оранжевый
            RGBColor(128, 0, 128), // фиолетовый
        ];

        for i in 0..N {
            let x_surf = self.r_surf[i];
            let prob: Vec<f64> = self.ioniz_prob[i].iter().map(|&p| p.into()).collect();

            // Берем цвет из палитры с циклическим повтором
            let color = colors[i % colors.len()].stroke_width(2);

            let series = LineSeries::new(
                t.iter().zip(prob.iter()).map(|(&t, &p)| (t, p)),
                color.clone(), // Клонируем ShapeStyle
            );

            chart
                .draw_series(series)
                .unwrap()
                .label(format!("x_surf = {:.2}", x_surf))
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
                });
        }

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .unwrap();
    }
}
