use super::{space::Xspace2D, wave_function::WaveFunction2D};
use crate::config::{C, F};
use crate::utils::hdf5_interface;
use ndarray::prelude::*;
use ndarray::Array1;
use plotters::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct DoubleIonizProb2e1d {
    pub r_surf: Vec<F>,
    pub ioniz_prob: Vec<Vec<F>>,
    pub t: Array1<F>,
}

impl DoubleIonizProb2e1d {
    /// r_surf = [8, 10, 15, 20] -- интеграл считается по области r_1,2>r_surf
    pub fn new(r_surf: Vec<F>, t: Array1<F>) -> Self {
        let ioniz_prob: Vec<Vec<F>> = vec![Vec::new(); r_surf.len()];
        Self {
            r_surf,
            ioniz_prob,
            t,
        }
    }

    pub fn save_as_hdf5(&self, path: &str) {
        let r_surf_arr = Array1::from_vec(self.r_surf.clone());
        hdf5_interface::write_to_hdf5(path, "r_surf", None, &r_surf_arr).unwrap();
        hdf5_interface::write_to_hdf5(path, "t", None, &self.t).unwrap();
        let n = self.r_surf.len();
        for i in 0..n {
            let r_surf_current: F = r_surf_arr[i];
            let ioniz_prob_i = Array1::from_vec(self.ioniz_prob[i].clone());
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
                "r_surf",
                format!("{r_surf_current}").as_str(),
            );
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

    fn calculate_ionization_probability(&self, wf: &WaveFunction2D, r_sq: F, dv: F) -> F {
        let x0 = wf.x.grid[0].clone();
        let x1 = wf.x.grid[1].clone();

        // Маски для 1 и второго электронов
        let calculate_mask =
            |x: &Array1<F>, r_sq: F| Array1::from_shape_fn((x.len()), |i| x[i] * x[i] > r_sq);

        let mask1 = calculate_mask(&x0, r_sq);
        let mask2 = calculate_mask(&x1, r_sq);

        // Параллельное суммирование с использованием масок
        let total_prob: F = (0..x0.len())
            .into_par_iter()
            .map(|i0| {
                let mut local_sum = 0.0;
                // Если условие для первого электрона не выполняется, пропускаем
                if mask1[i0] {
                    for i1 in 0..x1.len() {
                        // Если условие для второго электрона выполняется
                        if mask2[i1] {
                            local_sum += wf.psi[[i0, i1]].norm_sqr();
                        }
                    }
                }

                local_sum
            })
            .sum();

        total_prob * dv
    }

    pub fn add(&mut self, wf: &WaveFunction2D) {
        let dv = wf.x.dx[0] * wf.x.dx[1];

        for (i, &r_surf_val) in self.r_surf.iter().enumerate() {
            let r_sq = r_surf_val * r_surf_val;
            let ioniz_prob_current = self.calculate_ionization_probability(wf, r_sq, dv);
            self.ioniz_prob[i].push(ioniz_prob_current);
        }
    }

    pub fn plot(&self, output_path: &str) {
        let t = self.t.clone(); // костыль

        // Создаём область для рисования

        let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut y_max: F = 0.0;

        let N = self.r_surf.len();
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
