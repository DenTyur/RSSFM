use super::{space::Xspace1D, wave_function::WaveFunction1D};
use crate::common::tspace::Tspace;
use crate::macros::check_path;
use crate::traits::wave_function::WaveFunction;
use crate::utils::hdf5_interface;
use ndarray::prelude::*;
use ndarray::Array1;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::config::{C, F};

/// Отслеживает проекцию текущей волновой функции на начальное состояние
#[derive(Debug, Clone)]
pub struct ProjectionProb1D {
    psi0: Array1<C>, // начальное состояние
    dx: F,           // шаг сетки по x
    field_amplitude: F,
    pub times: Vec<F>, // моменты времени
    pub probs: Vec<F>, // |<psi0|psi(t)>|^2
    pub norms: Vec<F>, // |<psi0|psi(t)>|^2
}

impl ProjectionProb1D {
    /// Создаёт новый трекер на основе начальной волновой функции
    pub fn new(psi0: &WaveFunction1D, field_amplitude: F) -> Self {
        let dx = psi0.x.dx[0];
        Self {
            psi0: psi0.psi.clone(),
            dx,
            field_amplitude: field_amplitude,
            times: Vec::new(),
            probs: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// Добавляет текущее состояние в трекер
    pub fn add(&mut self, wf: &WaveFunction1D, t: F) {
        // Вычисляем скалярное произведение: sum(psi0^* * psi) * dx
        let overlap: C = self
            .psi0
            .iter()
            .zip(wf.psi.iter())
            .map(|(p0, p)| p0.conj() * p)
            .sum::<C>()
            * self.dx;

        let prob = overlap.norm_sqr(); // |overlap|^2
        self.times.push(t);
        self.probs.push(prob);
        self.norms.push(wf.prob_in_numerical_box());
    }

    /// Сохраняет времена и вероятности в HDF5 файл
    pub fn save_as_hdf5(&self, path: &str) {
        // Создаём массивы для записи
        let times_arr = Array1::from_vec(self.times.clone());
        let probs_arr = Array1::from_vec(self.probs.clone());
        let norms_arr = Array1::from_vec(self.norms.clone());

        hdf5_interface::write_to_hdf5(path, "time", None, &times_arr)
            .expect("Failed to write time");
        hdf5_interface::write_to_hdf5(path, "probability", None, &probs_arr)
            .expect("Failed to write probability");

        hdf5_interface::write_to_hdf5(path, "norm", None, &norms_arr)
            .expect("Failed to write norm");
        let field_ampl = Some(self.field_amplitude);

        hdf5_interface::write_scalar_to_hdf5(path, "field_amplitude", None, self.field_amplitude)
            .expect("Failed to write field_amplitude");

        // Добавим атрибут с информацией о типе данных
        hdf5_interface::add_str_group_attr(
            path,
            "/",
            "description",
            "Projection probability |<psi0|psi(t)>|^2",
        )
        .ok();
    }
}

#[derive(Debug, Clone)]
pub struct IonizProb1D {
    pub x_surf: Vec<F>,
    pub ioniz_prob: Vec<Vec<F>>,
    // pub x: Xspace1D,
    pub t: Array1<F>,
}

impl IonizProb1D {
    pub fn new(x_surf: Vec<F>, t: Array1<F>) -> Self {
        let ioniz_prob: Vec<Vec<F>> = vec![Vec::new(); x_surf.len()];
        Self {
            x_surf,
            ioniz_prob,
            t,
        }
    }

    pub fn save_as_hdf5(&self, path: &str) {
        check_path!(path);
        let x_surf_arr = Array1::from_vec(self.x_surf.to_vec());
        hdf5_interface::write_to_hdf5(path, "x_surf", None, &x_surf_arr).unwrap();
        hdf5_interface::write_to_hdf5(path, "t", None, &self.t).unwrap();
        let n = self.x_surf.len();
        for i in 0..n {
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

    pub fn add(&mut self, wf: &WaveFunction1D) {
        let n = self.x_surf.len();
        for i in 0..n {
            let xmin = wf.x.grid[0][[0]];
            let ds = wf.x.dx[0];
            let indx0 = ((self.x_surf[i] - xmin) / ds).round() as usize;
            let psi_slice = wf.psi.slice(s![indx0..]);
            self.ioniz_prob[i].push(psi_slice.mapv(|c| c.norm_sqr()).sum() * wf.x.dx[0]);
        }
    }

    pub fn plot(&self, output_path: &str) {
        // длина векторов
        let n = self.x_surf.len();
        // Создаём область для рисования
        let t = self.t.clone();

        let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut y_max: F = 0.0;
        for i in 0..n {
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
            .build_cartesian_2d(t[0]..t[t.len() - 1].into(), 0.0f64..y_max as f64 * 1.1)
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

        for i in 0..n {
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
