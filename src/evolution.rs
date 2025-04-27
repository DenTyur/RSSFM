use crate::field;
use crate::gauge;
use crate::parameters;
use crate::potentials;
use crate::wave_function;
use field::Field2D;
use gauge::{LenthGauge, VelocityGauge};
use itertools::multizip;
use ndarray::prelude::*;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use num_complex::Complex;
use parameters::*;
use potentials::{br_1e2d_external, AtomicPotential};
use rayon::prelude::*;
use std::f32::consts::PI;
use wave_function::WaveFunction;
// Чем более непонятным становится код, тем он круче! (с)
// Всратость не есть недостаток!

type F = f32;
type C = Complex<f32>;
const I: C = Complex::I;

pub trait EvolutionSSFM {
    // Гауге:)
    // От калибровки зависят только эволюции в импульсном и координатном пространстве
    fn x_evol_half(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace);
    fn x_evol(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace);
    fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace);
}

impl<'a> EvolutionSSFM for VelocityGauge<'a> {
    fn x_evol_half(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
        // эволюция в координатном пространстве на половину временного шага
        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            // Соединяем первый индекс psi с x
            .par_bridge() // по нулевой оси делаем цикл параллельным
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter()))
                    // Соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // Решил, что лучше вычислять по факту потенциал, а не хранить его массив в
                        // памяти. Меньше места занимает, а скорость такая же.
                        let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
                        // эволюция psi
                        *psi_elem *= (-I * 0.5 * t.dt * atomic_potential_elem).exp();
                    });
            });
    }

    fn x_evol(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
        // эволюция в координатном пространстве на полный временной шаг
        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            // Соединяем первый индекс psi с x
            .par_bridge() // по нулевой оси делаем цикл параллельным
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter()))
                    // Соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // Решил, что лучше вычислять по факту потенциал, а не хранить его массив в
                        // памяти. Меньше места занимает, а скорость такая же.
                        let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
                        // эволюция psi
                        *psi_elem *= (-I * t.dt * atomic_potential_elem).exp();
                    });
            });
    }

    fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace) {
        // эволюция в импульсном пространстве
        let vec_pot = self.field.vec_pot(t.current);

        multizip((psi.psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
            // соединяем первый индекс psi_p с px
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(p.grid[1].iter())
                    // соединяем второй индекс psi_p с py
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I
                            * t.dt
                            * (0.5 * px * px + 0.5 * py * py + vec_pot[0] * px + vec_pot[1] * py))
                            .exp();
                    });
            });
    }
}

impl<'a> EvolutionSSFM for LenthGauge<'a> {
    fn x_evol_half(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
        // эволюция в координатном пространстве на половину временного шага

        let electric_field = self.field.electric_field_time_dependence(t.current);

        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            // Соединяем первый индекс psi с x
            .par_bridge() // по нулевой оси делаем цикл параллельным
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter()))
                    // Соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // Решил, что лучше вычислять по факту потенциал, а не хранить его массив в
                        // памяти. Меньше места занимает, а скорость такая же.
                        let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
                        // эволюция psi
                        *psi_elem *= (-I
                            * 0.5
                            * t.dt
                            * (atomic_potential_elem
                                + electric_field[0] * x_point // заряд у электрона минус, поэтому
                            // здесь плюс
                                + electric_field[1] * y_point))
                            .exp();
                    });
            });
    }

    fn x_evol(&self, psi: &mut WaveFunction, x: &Xspace, t: &Tspace) {
        // эволюция в координатном пространстве на полный временной шаг
        let electric_field = self.field.electric_field_time_dependence(t.current);

        multizip((psi.psi.axis_iter_mut(Axis(0)), x.grid[0].iter()))
            // Соединяем первый индекс psi с x
            .par_bridge() // по нулевой оси делаем цикл параллельным
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), x.grid[1].iter()))
                    // Соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // Решил, что лучше вычислять по факту потенциал, а не хранить его массив в
                        // памяти. Меньше места занимает, а скорость такая же.
                        let atomic_potential_elem = br_1e2d_external(*x_point, *y_point);
                        // эволюция psi
                        *psi_elem *= (-I
                            * t.dt
                            * (atomic_potential_elem
                                + electric_field[0] * x_point // заряд у электрона минус, поэтому
                            // здесь плюс
                                + electric_field[1] * y_point))
                            .exp();
                    });
            });
    }

    fn p_evol(&self, psi: &mut WaveFunction, p: &Pspace, t: &Tspace) {
        // эволюция в импульсном пространстве
        multizip((psi.psi.axis_iter_mut(Axis(0)), p.grid[0].iter()))
            // соединяем первый индекс psi_p с px
            .par_bridge()
            .for_each(|(mut psi_row, px)| {
                psi_row
                    .iter_mut()
                    .zip(p.grid[1].iter())
                    // соединяем второй индекс psi_p с py
                    .for_each(|(psi_elem, py)| {
                        *psi_elem *= (-I * t.dt * (0.5 * px * px + 0.5 * py * py)).exp();
                    });
            });
    }
}

pub struct SSFM<'a, G: EvolutionSSFM> {
    gauge: &'a G,
    fft: FftMaker2d,
    x: &'a Xspace,
    p: &'a Pspace,
}

impl<'a, G: EvolutionSSFM> SSFM<'a, G> {
    pub fn new(gauge: &'a G, x: &'a Xspace, p: &'a Pspace) -> Self {
        let fft = FftMaker2d::new(&x.n);
        Self { gauge, fft, x, p }
    }

    pub fn demodify_psi(&mut self, psi: &mut WaveFunction) {
        // демодифицирует "psi для DFT" обратно в psi
        multizip((psi.psi.axis_iter_mut(Axis(0)), self.x.grid[0].iter()))
            // соединяем первый индекс psi с x
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), self.x.grid[1].iter()))
                    // соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // демодифицируем psi
                        *psi_elem *= (2. * PI) / (self.x.dx[0] * self.x.dx[1])
                            * (I * (self.p.p0[0] * x_point + self.p.p0[1] * y_point)).exp();
                    });
            });
    }

    pub fn modify_psi(&mut self, psi: &mut WaveFunction) {
        // модифицирует psi для FFT
        multizip((psi.psi.axis_iter_mut(Axis(0)), self.x.grid[0].iter()))
            // соединяем первый индекс psi с x
            .par_bridge()
            .for_each(|(mut psi_row, x_point)| {
                multizip((psi_row.iter_mut(), self.x.grid[1].iter()))
                    // соединяем второй индекс psi с y
                    .for_each(|(psi_elem, y_point)| {
                        // модифицируем psi
                        *psi_elem *= self.x.dx[0] * self.x.dx[1] / (2. * PI)
                            * (-I * (self.p.p0[0] * x_point + self.p.p0[1] * *y_point)).exp();
                    });
            });
    }

    pub fn time_step_evol(
        &mut self,
        psi: &mut WaveFunction,
        t: &mut Tspace,
        psi_x_save_path: Option<&str>,
        psi_p_save_path: Option<&str>,
    ) {
        if let Some(path) = psi_x_save_path {
            psi.save_psi(path).unwrap();
        }
        self.modify_psi(psi);
        self.gauge.x_evol_half(psi, self.x, t);

        for _i in 0..t.n_steps - 1 {
            self.fft.do_fft(psi);
            // Можно оптимизировать p_evol
            self.gauge.p_evol(psi, self.p, t);
            self.fft.do_ifft(psi);
            self.gauge.x_evol(psi, self.x, t);
            t.current += t.dt;
        }

        self.fft.do_fft(psi);
        self.gauge.p_evol(psi, self.p, t);
        if let Some(path) = psi_p_save_path {
            psi.save_psi(path).unwrap();
        }
        self.fft.do_ifft(psi);
        self.gauge.x_evol_half(psi, self.x, t);
        self.demodify_psi(psi);
        t.current += t.dt;
    }
}

pub struct FftMaker2d {
    pub handler: Vec<FftHandler<F>>,
    pub psi_temp: Array2<C>,
}

impl FftMaker2d {
    pub fn new(n: &Vec<usize>) -> Self {
        Self {
            handler: Vec::from_iter(0..n.len()) // тоже костыль! как это сделать через функцию?
                .iter()
                .map(|&i| FftHandler::new(n[i]))
                .collect(),
            psi_temp: Array::zeros((n[0], n[1])),
        }
    }

    pub fn do_fft(&mut self, psi: &mut WaveFunction) {
        ndfft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[0], 0);
        ndfft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[1], 1);
    }
    pub fn do_ifft(&mut self, psi: &mut WaveFunction) {
        ndifft_par(&psi.psi, &mut self.psi_temp, &mut self.handler[1], 1);
        ndifft_par(&self.psi_temp, &mut psi.psi, &mut self.handler[0], 0);
    }
}
