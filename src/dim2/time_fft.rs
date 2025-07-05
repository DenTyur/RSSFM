use crate::common::tspace::Tspace;
use crate::config::{C, F, PI};
use crate::dim1::fft_maker::FftMaker1D;
use crate::dim2::{
    fft_maker::FftMaker2D, gauge::LenthGauge2D, space::Xspace2D, ssfm::SSFM2D,
    wave_function::WaveFunction2D,
};
use crate::measure_time;
use crate::potentials::absorbing_potentials::absorbing_potential_1d;
use crate::potentials::potentials;
use crate::traits::fft_maker::{self, FftMaker};
use crate::traits::{
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use ndarray::prelude::*;
use plotters::coord::Shift;
use plotters::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

// плохо написано, но работает:)
pub struct TimeFFT {
    t: Tspace,
    point: [F; 2],
    ind_point: [usize; 2],
    psi_in_point: Vec<C>,
    pub energy: Array1<F>,
    psi_fft: Array1<C>,
}

impl TimeFFT {
    const AU_TO_EV: F = 4.3597e-11 / 1.60217733e-12;
    pub fn new(t: Tspace, point: [F; 2], x: &Xspace2D) -> Self {
        let x_min = x.grid[0][[0]];
        let y_min = x.grid[1][[0]];
        let ind_point_x = ((point[0] - x_min) / x.dx[0]).round() as usize;
        let ind_point_y = ((point[1] - y_min) / x.dx[1]).round() as usize;
        let psi_in_point: Vec<C> = Vec::new();
        let energy_step = 2.0 * PI / (t.nt as F * t.t_step()) * Self::AU_TO_EV;
        // ????????????????????????????????????????
        let energy_min = -PI / t.t_step() * Self::AU_TO_EV + energy_step;
        let energy: Array1<F> = Array::range(
            energy_min,
            energy_min + energy_step * t.nt as F,
            energy_step,
        );
        let psi_fft = Array1::<C>::zeros(t.nt);
        Self {
            t,
            point,
            ind_point: [ind_point_x, ind_point_y],
            psi_in_point,
            energy,
            psi_fft,
        }
    }
    pub fn add_psi_in_point(&mut self, wf: &WaveFunction2D) {
        self.psi_in_point
            .push(wf.psi[(self.ind_point[0], self.ind_point[1])]);
    }

    pub fn shift(&mut self) {
        let len = self.psi_fft.len();
        let mid = len / 2;
        let mut shifted = Array1::<C>::zeros(len);

        // Первая половина -> в конец
        shifted
            .slice_mut(s![..len - mid])
            .assign(&self.psi_fft.slice(s![mid..]));
        // Вторая половина -> в начало
        shifted
            .slice_mut(s![len - mid..])
            .assign(&self.psi_fft.slice(s![..mid]));
        self.psi_fft = shifted;
    }

    // переписать!!!
    pub fn compute_spectrum(&mut self) {
        let n = self.t.nt;
        self.psi_fft = Array::from_vec(self.psi_in_point.clone());
        let mut fft_maker = FftMaker1D::new(&[n]);
        fft_maker.fft(&mut self.psi_fft);
        self.shift(); // Теперь вызывается как метод структуры
        let reversed = self.psi_fft.slice(s![..;-1]).to_owned();
        self.psi_fft.assign(&reversed);
    }

    pub fn find_energy_max(&self) -> F {
        let n = self.t.nt;
        let mut psi_fft_norm_sq: Array1<F> = Array::zeros(n);

        self.psi_fft
            .iter()
            .zip(psi_fft_norm_sq.iter_mut())
            .par_bridge()
            .for_each(|(psi_elem, a_elem)| {
                *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
            });

        let (max_index, max_value) = psi_fft_norm_sq
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &val)| (i, val))
            .expect("Массив не должен быть пустым!");
        self.energy[[max_index]]
    }

    pub fn plot_log(&self, file_path: &str) {
        let x_values = self.energy.clone();
        let mut psi_norm_sq: Array1<F> = Array::zeros(self.t.nt);

        self.psi_fft
            .iter()
            .zip(psi_norm_sq.iter_mut())
            .par_bridge()
            .for_each(|(psi_elem, a_elem)| {
                *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
            });

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

        let emax = self.find_energy_max();
        let de = self.energy[[1]] - self.energy[[0]];

        // Создаем график с линейной осью x и логарифмической осью y
        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("E = {emax} eV, dE = {de} eV").as_str(),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        // Настраиваем ось y как логарифмическую
        chart
            .configure_mesh()
            .x_desc("energy [eV]")
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
}
