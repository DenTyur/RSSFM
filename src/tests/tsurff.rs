use super::super::tsurff::*;
use rayon::prelude::*;

use super::gauss::Gauss;
use crate::field;
use crate::flow;
use crate::gauge;
use crate::heatmap;
use crate::macros::measure_time;
use crate::parameters;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
use flow::{Circle, Flow, Flux, Square, SurfaceFlow};
use gauge::{LenthGauge, VelocityGauge};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::marker::{Send, Sync};
use std::time::Instant;
use volkov::{Volkov, VolkovGauge};
use wave_function::{ValueAndSpaceDerivatives, WaveFunction};

type F = f32;
type C = Complex<F>;
const I: C = Complex::I;

fn set_psi_gauss(psi: &mut Array2<C>, gauss: &Gauss, x: &Xspace, t: F) {
    psi.axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut psi_row, x_point)| {
            psi_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(psi_elem, y_point)| {
                    *psi_elem = gauss.wf([*x_point, *y_point], t);
                })
        });
}

/// Вычисляет импульсное распределение для
/// гауссового пакета, заданного аналитически
#[test]
fn tsurff_gauss_analit() {
    let x_dir_path = "src/arrays_saved";
    let psi_path = "src/arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 1, 75);

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 2);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);

    //гауссов пакет
    let gauss = Gauss::new([4.0, 0.0], 1.0);

    // генерация начальной волновой функции psi_initial
    let mut psi_initial: Array2<C> = Array::zeros((x.n[0], x.n[1]));
    set_psi_gauss(&mut psi_initial, &gauss, &x, 0.0);

    // структура для волновой функции
    let mut psi = WaveFunction::new(psi_initial, &x);
    psi.normalization_by_1();

    //строим аналитическое импульсное распределение
    gauss.plot_momentum_distribution(&p, "src/tests/out/tsurff/analit_moment_distr.png");

    // инициализируем внешнее поле
    let field = Field2D {
        amplitude: 0.0,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    // указываем калибровку поля
    let gauge = VelocityGauge::new(&field);

    // создаем структуру для потока вероятности через поверхность
    let surface = Square::new(30.0, &x);
    let mut flow = Flow::new(&gauge, &surface);

    // t-surff
    let mut tsurff = Tsurff::new(&gauge, &surface, &x, &p, Some(p.grid[0][p.n[0] - 1]));
    // let mut examine_tsurff = Tsurff::new(&gauge, &surface, &x, &p);
    // let mut instance_psi_p: Array<C, Ix2> = Array::zeros((p.n[0], p.n[1]));

    // временная эволюция
    let total_time = Instant::now();
    for i in 0..t.nt {
        let time_step = Instant::now();
        println!(
            "STEP {}/{}, t.current={:.5}, norm = {}, prob_in_box = {}",
            i,
            t.nt,
            t.current,
            psi.norm(),
            psi.prob_in_numerical_box(),
        );
        //============================================================
        //                       EVOLUTION
        //============================================================
        measure_time!("evol", {
            set_psi_gauss(&mut psi.psi, &gauss, &x, t.current);
        });
        //обновление производных
        measure_time!("update_deriv", {
            psi.update_derivatives();
            // psi.dpsi_dx = gauss.dwf_dx_as_array(&x, t.current);
            // psi.dpsi_dy = gauss.dwf_dy_as_array(&x, t.current);
        });
        //============================================================
        //                       t-SURFF
        //============================================================
        measure_time!("t-surff", {
            tsurff.time_integration_step(&psi, &t);
        });

        //============================================================
        //                    EXAMINE t-SURFF
        //============================================================
        // measure_time!("examine t-surff", {
        //     let x_border: F = surface.border;
        //     let ind_x_border: usize = ((x_border - x.x0[0]) / x.dx[0]).round() as usize;
        //     let ind_x_minus_border: usize = ((-x_border - x.x0[0]) / x.dx[0]).round() as usize;
        //
        //     // интеграл по поверхности
        //     let instance_surf_flow = |volkov: &Volkov<VelocityGauge>, psi: &WaveFunction, t: F| {
        //         let flow: C = (ind_x_minus_border..ind_x_border) // итерируем по y
        //             .into_par_iter()
        //             .map(|i| {
        //                 let ind = [ind_x_border, i];
        //                 let x_point = x.grid[0][ind[0]];
        //                 let y_point = x.grid[1][ind[1]];
        //                 let point = [x_point, y_point];
        //                 let psi_val = psi.psi[ind];
        //                 let dpsi_dx = gauss.dwf_dx(point, t);
        //                 let volkov_val = volkov.value(point);
        //                 let dvolkov_dx = volkov.deriv(point)[0];
        //                 I / 2.0 * (psi_val * dvolkov_dx.conj() - volkov_val.conj() * dpsi_dx)
        //             })
        //             .sum::<C>()
        //             * x.dx[1];
        //         flow
        //     };
        //
        //     // интеграл по времени
        //     examine_tsurff
        //         .psi_p
        //         .axis_iter_mut(Axis(0))
        //         .zip(p.grid[0].iter())
        //         .par_bridge()
        //         .for_each(|(mut psi_p_row, px)| {
        //             psi_p_row
        //                 .iter_mut()
        //                 .zip(p.grid[1].iter())
        //                 .for_each(|(psi_p_elem, py)| {
        //                     let volkov = Volkov::new(&gauge, [*px, *py], t.current);
        //                     *psi_p_elem +=
        //                         instance_surf_flow(&volkov, &psi, t.current) * t.t_step();
        //                 })
        //         });
        // });
        //==============Сохранение картинок===========================
        psi.plot(
            format!("src/tests/out/tsurff/psi_x/psi_x_{i}.png").as_str(),
            [0.0, 1e-6],
        );
        tsurff.plot(format!("src/tests/out/tsurff/tsurff/pwf_tsurff{i}.png").as_str());
        // examine_tsurff
        // .plot(format!("src/tests/out/tsurff/examine_tsurff/pwf_tsurff{i}.png").as_str());
        //============================================================
        //                       Flow
        //============================================================
        measure_time!("flow_time", {
            flow.add_instance_flow(&psi, t.current);
        });

        //============================================================
        println!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        );

        //==========
        t.current += t.t_step();
        //==========
    }

    //вывод полного потока
    let total_flow = flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
    // график потока
    flow.plot_flow("src/tests/out/tsurff/flow/flow_graph.png");
}

#[test]
fn tsurff_gauss_ssfm() {
    let x_dir_path = "src/arrays_saved";
    let psi_path = "src/arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 1, 75);

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 2);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);

    //гауссов пакет
    let gauss = Gauss::new([4.0, 0.0], 1.0);

    // генерация начальной волновой функции psi_initial
    let mut psi_initial: Array2<C> = Array::zeros((x.n[0], x.n[1]));
    set_psi_gauss(&mut psi_initial, &gauss, &x, 0.0);

    // структура для волновой функции
    let mut psi = WaveFunction::new(psi_initial, &x);
    psi.normalization_by_1();

    //строим аналитическое импульсное распределение
    gauss.plot_momentum_distribution(&p, "src/tests/out/tsurff/analit_moment_distr.png");

    // инициализируем внешнее поле
    let field = Field2D {
        amplitude: 0.0,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    // указываем калибровку поля
    let gauge = VelocityGauge::new(&field);

    // создаем структуру для потока вероятности через поверхность
    let surface = Square::new(30.0, &x);
    let mut flow = Flow::new(&gauge, &surface);

    // t-surff
    let mut tsurff = Tsurff::new(&gauge, &surface, &x, &p, Some(p.grid[0][p.n[0] - 1]));
    // let mut examine_tsurff = Tsurff::new(&gauge, &surface, &x, &p);
    // let mut instance_psi_p: Array<C, Ix2> = Array::zeros((p.n[0], p.n[1]));

    // временная эволюция
    let total_time = Instant::now();
    for i in 0..t.nt {
        let time_step = Instant::now();
        println!(
            "STEP {}/{}, t.current={:.5}, norm = {}, prob_in_box = {}",
            i,
            t.nt,
            t.current,
            psi.norm(),
            psi.prob_in_numerical_box(),
        );
        //============================================================
        //                       EVOLUTION
        //============================================================
        measure_time!("evol", {
            set_psi_gauss(&mut psi.psi, &gauss, &x, t.current);
        });
        //обновление производных
        measure_time!("update_deriv", {
            psi.update_derivatives();
            // psi.dpsi_dx = gauss.dwf_dx_as_array(&x, t.current);
            // psi.dpsi_dy = gauss.dwf_dy_as_array(&x, t.current);
        });
        //============================================================
        //                       t-SURFF
        //============================================================
        measure_time!("t-surff", {
            tsurff.time_integration_step(&psi, &t);
        });

        //============================================================
        //                    EXAMINE t-SURFF
        //============================================================
        // measure_time!("examine t-surff", {
        //     let x_border: F = surface.border;
        //     let ind_x_border: usize = ((x_border - x.x0[0]) / x.dx[0]).round() as usize;
        //     let ind_x_minus_border: usize = ((-x_border - x.x0[0]) / x.dx[0]).round() as usize;
        //
        //     // интеграл по поверхности
        //     let instance_surf_flow = |volkov: &Volkov<VelocityGauge>, psi: &WaveFunction, t: F| {
        //         let flow: C = (ind_x_minus_border..ind_x_border) // итерируем по y
        //             .into_par_iter()
        //             .map(|i| {
        //                 let ind = [ind_x_border, i];
        //                 let x_point = x.grid[0][ind[0]];
        //                 let y_point = x.grid[1][ind[1]];
        //                 let point = [x_point, y_point];
        //                 let psi_val = psi.psi[ind];
        //                 let dpsi_dx = gauss.dwf_dx(point, t);
        //                 let volkov_val = volkov.value(point);
        //                 let dvolkov_dx = volkov.deriv(point)[0];
        //                 I / 2.0 * (psi_val * dvolkov_dx.conj() - volkov_val.conj() * dpsi_dx)
        //             })
        //             .sum::<C>()
        //             * x.dx[1];
        //         flow
        //     };
        //
        //     // интеграл по времени
        //     examine_tsurff
        //         .psi_p
        //         .axis_iter_mut(Axis(0))
        //         .zip(p.grid[0].iter())
        //         .par_bridge()
        //         .for_each(|(mut psi_p_row, px)| {
        //             psi_p_row
        //                 .iter_mut()
        //                 .zip(p.grid[1].iter())
        //                 .for_each(|(psi_p_elem, py)| {
        //                     let volkov = Volkov::new(&gauge, [*px, *py], t.current);
        //                     *psi_p_elem +=
        //                         instance_surf_flow(&volkov, &psi, t.current) * t.t_step();
        //                 })
        //         });
        // });
        //==============Сохранение картинок===========================
        psi.plot(
            format!("src/tests/out/tsurff/psi_x/psi_x_{i}.png").as_str(),
            [0.0, 1e-6],
        );
        tsurff.plot(format!("src/tests/out/tsurff/tsurff/pwf_tsurff{i}.png").as_str());
        // examine_tsurff
        // .plot(format!("src/tests/out/tsurff/examine_tsurff/pwf_tsurff{i}.png").as_str());
        //============================================================
        //                       Flow
        //============================================================
        measure_time!("flow_time", {
            flow.add_instance_flow(&psi, t.current);
        });

        //============================================================
        println!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        );

        //==========
        t.current += t.t_step();
        //==========
    }

    //вывод полного потока
    let total_flow = flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
    // график потока
    flow.plot_flow("src/tests/out/tsurff/flow/flow_graph.png");
}
