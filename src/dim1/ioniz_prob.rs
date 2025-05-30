use super::{
    fft_maker::FftMaker1D,
    space::{Pspace1D, Xspace1D},
    wave_function::WaveFunction1D,
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
pub struct IonizProb1D<const N: usize> {
    pub x_surf: [F; N],
    pub ioniz_prob: [Vec<F>; N],
    pub x: Xspace1D,
    pub t: Tspace,
}

impl<const N: usize> IonizProb1D<N> {
    pub fn new(x_surf: [F; N], x: Xspace1D, t: Tspace) -> Self {
        let ioniz_prob: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
        Self {
            x_surf,
            ioniz_prob,
            x,
            t,
        }
    }

    pub fn save_as_hdf5(&self, path: &str) {
        let x_surf_arr = Array1::from_vec(self.x_surf.to_vec());
        hdf5_interface::write_to_hdf5(path, "x_surf", None, &x_surf_arr).unwrap();
        hdf5_interface::write_to_hdf5(path, "t", None, &self.t.grid).unwrap();
        for i in 0..N {
            let ioniz_prob_i = Array1::from_vec(self.ioniz_prob[i].to_vec());
            hdf5_interface::write_to_hdf5(
                path,
                format!("ioniz_prob_{i}").as_str(),
                None,
                &ioniz_prob_i,
            )
            .unwrap();
        }
    }

    pub fn add(&mut self, wf: &WaveFunction1D) {
        for i in 0..N {
            let xmin = self.x.grid[0][[0]];
            let ds = self.x.dx[0];
            let indx0 = ((self.x_surf[i] - xmin) / ds).round() as usize;
            let psi_slice = wf.psi.slice(s![indx0..]);
            self.ioniz_prob[i].push(psi_slice.mapv(|c| c.norm_sqr()).sum() * self.x.dx[0]);
        }
    }

    pub fn plot(&self, tspace: &Tspace, output_path: &str) {
        // Создаём область для рисования
        let t = tspace.grid.clone();

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
            let x_surf = self.x_surf[i];
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
