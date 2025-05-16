use super::logcolormap;
use crate::config::{C, F};
use crate::macros::check_path;
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use rayon::prelude::*;
use std::fs::File;

/// Структура для вычисления импульсного распределения двукратно ионизированных ионов
/// для невзаимодействующих электронов.
/// Сейчас предполагается, что импульсная сетка по всем осям одинаковая.
/// Можно модифицировать на общий случай разных сеток.
/// dimd -- Double Ion Momentum Distribution
pub struct DIMD2D {
    psi1: Array2<C>,
    psi2: Array2<C>,
    dimd: Array2<F>,
    p: [Array1<F>; 2],
    p_ion: [Array1<F>; 2],
}

impl DIMD2D {
    pub fn new(psi1_path: &str, psi2_path: &str, p_path: &str) -> Self {
        let reader1 = File::open(psi1_path).unwrap();
        let psi1 = Array::<C, Ix2>::read_npy(reader1).unwrap();

        let reader2 = File::open(psi2_path).unwrap();
        let psi2 = Array::<C, Ix2>::read_npy(reader2).unwrap();

        let readerp = File::open(p_path).unwrap();
        let p1 = Array::<F, Ix1>::read_npy(readerp).unwrap();
        let n1: usize = p1.len();
        let dp1 = p1[1] - p1[0];
        let p_ion_min = p1[0] * 2.0;
        let p_ion_max = p1[n1 - 1] * 2.0;
        let p_ion1: Array1<F> = Array::range(p_ion_min, p_ion_max + dp1, dp1);
        let n_ion = p_ion1.len();
        let dimd: Array2<F> = Array::zeros((n_ion, n_ion));

        Self {
            psi1,
            psi2,
            dimd,
            p: [p1.clone(), p1.clone()],
            p_ion: [p_ion1.clone(), p_ion1.clone()],
        }
    }

    pub fn compute_dimd(&mut self) {
        let np1 = self.p[0].len() - 1;
        let np2 = self.p[1].len() - 1;
        let dp1 = self.p[0][1] - self.p[0][0];
        let dp2 = self.p[1][1] - self.p[1][0];
        self.dimd
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(n1_plus_n2, mut row)| {
                for m1_plus_m2 in 0..(np2 * 2) {
                    let mut sum = 0.0;

                    // Вычисляем границы для оптимизации итераций
                    let n1_start = n1_plus_n2.saturating_sub(np1 - 1);
                    let n1_end = (n1_plus_n2).min(np1 - 1);

                    let m1_start = m1_plus_m2.saturating_sub(np2 - 1);
                    let m1_end = (m1_plus_m2).min(np2 - 1);

                    // Внутренние циклы
                    for n1 in n1_start..=n1_end {
                        let n2 = n1_plus_n2 - n1;
                        for m1 in m1_start..=m1_end {
                            let m2 = m1_plus_m2 - m1;
                            sum += self.psi1[[n1, m1]].norm().powi(2)
                                * self.psi2[[n2, m2]].norm().powi(2);
                        }
                    }

                    row[m1_plus_m2] = sum * dp1 * dp2;
                }
            });
    }

    pub fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &self.dimd,
            &self.p_ion[0],
            &self.p_ion[1],
            (colorbar_limits[0], colorbar_limits[1]),
            path,
        )
        .unwrap();
    }
}
