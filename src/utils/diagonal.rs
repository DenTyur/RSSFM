use crate::config::F;
use crate::dim4::probability_density_2d::ProbabilityDensity2D;
use crate::dim4::wave_function::WaveFunction4D;
use crate::measure_time;
use crate::traits::wave_function::WaveFunction;
use crate::utils::processing::create_zeroed_wavefunction;
use ndarray::prelude::*;
pub struct Diagonal {
    diag: Array1<F>,
    axis: Array1<F>,
}
impl Diagonal {
    pub fn init_from(array2d: &Array2<F>, x: &Array1<F>) -> Self {
        assert_eq!(
            array2d.nrows(),
            array2d.ncols(),
            "Матрица должна быть квадратной"
        );
        assert_eq!(
            array2d.nrows(),
            x.len(),
            "Размеры массива x и матрицы должны совпадать"
        );

        let diag = Array1::from_shape_fn(array2d.nrows(), |i| array2d[[i, i]]);

        Self {
            diag,
            axis: x.clone(),
        }
    }

    /// Возвращает массив индексов локальных максимумов
    pub fn find_local_maxima(&self) -> Array1<usize> {
        let n = self.diag.len();

        if n < 2 {
            return Array1::default(0); // Недостаточно точек для поиска максимумов
        }

        let mut maxima_indices = Vec::new();

        // Проверяем первую точку
        if self.diag[0] > self.diag[1] {
            maxima_indices.push(0);
        }

        // Проверяем внутренние точки
        for i in 1..n - 1 {
            if self.diag[i] > self.diag[i - 1] && self.diag[i] > self.diag[i + 1] {
                maxima_indices.push(i);
            }
        }

        // Проверяем последнюю точку
        if self.diag[n - 1] > self.diag[n - 2] {
            maxima_indices.push(n - 1);
        }

        Array1::from_vec(maxima_indices)
    }

    /// Находит локальные максимумы и возвращает их значения и координаты
    /// Возвращает tuple: (индексы, значения на диагонали, значения на оси x)
    pub fn get_local_maxima_above(&self, threshold: F) -> (Array1<usize>, Array1<F>, Array1<F>) {
        let indices = self.find_local_maxima_above(threshold);
        let values = Array1::from_shape_fn(indices.len(), |i| self.diag[indices[i]]);
        let x_values = Array1::from_shape_fn(indices.len(), |i| self.axis[indices[i]]);

        (indices, values, x_values)
    }

    /// Находит локальные максимумы с порогом (только значения выше threshold)
    pub fn find_local_maxima_above(&self, threshold: F) -> Array1<usize> {
        let indices = self.find_local_maxima();
        indices
            .into_iter()
            .filter(|&i| self.diag[i] > threshold)
            .collect()
    }
}
