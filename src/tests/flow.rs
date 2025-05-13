use super::super::flow::*;
use super::gauss;
use super::gauss::Gauss;
use crate::field;
use crate::gauge;
use crate::heatmap;
use crate::parameters;
use crate::volkov;
use crate::wave_function;
use field::Field2D;
use gauge::{LenthGauge, VelocityGauge};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::*;
use rayon::prelude::*;
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

fn set_flux(gauge: &VelocityGauge, flux: &mut Array2<F>, psi: &WaveFunction, x: &Xspace, t: F) {
    flux.axis_iter_mut(Axis(0))
        .zip(x.grid[0].iter())
        .par_bridge()
        .for_each(|(mut flux_row, x_point)| {
            flux_row
                .iter_mut()
                .zip(x.grid[1].iter())
                .for_each(|(flux_elem, y_point)| {
                    *flux_elem = gauge.compute_flux([*x_point, *y_point], psi, psi, t)[0].re;
                })
        });
}

/// Строит плотность потока вероятности гауссового пакета
#[test]
fn flux_gauss() {
    let x_dir_path = "src/arrays_saved";
    let psi_path = "src/arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.1, 1, 150);

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
    let mut examine_flow = Flow::new(&gauge, &surface);
    let mut analit_flow = Flow::new(&gauge, &surface);

    // создаем массив для плотности потока вероятности
    let mut flux: Array2<F> = Array::zeros((x.n[0], x.n[1]));

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
        let time_evol = Instant::now();
        set_psi_gauss(&mut psi.psi, &gauss, &x, t.current);
        println!("evol = {:.3}", time_evol.elapsed().as_secs_f32());
        //обновление производных
        psi.update_derivatives();
        //=========Сохранение картинок===========================
        psi.plot(
            format!("src/tests/out/flow/psi_x/psi_x_{i}.png").as_str(),
            [0.0, 1e-6],
        );
        //=========Сохранение вф===========================
        psi.save_psi(format!("src/tests/out/flow/psi_x_arr/psi_x_{i}.npy").as_str())
            .unwrap();
        //============================================================
        //                       Flow
        //============================================================
        let flow_time = Instant::now();
        flow.add_instance_flow(&psi, t.current);
        println!("flow_time = {}", flow_time.elapsed().as_secs_f32());
        //============================================================
        //                       Flux
        //============================================================
        flux.axis_iter_mut(Axis(0))
            .zip(x.grid[0].iter())
            .par_bridge()
            .for_each(|(mut flux_row, x_point)| {
                flux_row
                    .iter_mut()
                    .zip(x.grid[1].iter())
                    .for_each(|(flux_elem, y_point)| {
                        if *x_point > x.x0[0] && *x_point < x.grid[0][x.n[0] - 1] {
                            if *y_point > x.x0[1] && *y_point < x.grid[1][x.n[1] - 1] {
                                *flux_elem =
                                    gauge.compute_flux([*x_point, *y_point], &psi, &psi, t.current)
                                        [1]
                                    .re
                                    .abs();
                            }
                        }
                    })
            });
        let (size_x, size_y, size_colorbar) = (500, 500, 60);
        let (colorbar_min, colorbar_max) = (1e-5, 1e-2);

        heatmap::plot_heatmap(
            &x.grid[0],
            &x.grid[1],
            &flux,
            size_x,
            size_y,
            size_colorbar,
            colorbar_min,
            colorbar_max,
            format!("src/tests/out/flow/flux/flux{i}.png").as_str(),
        );
        //============================================================
        //                Examine Flow Square
        //============================================================
        let x_border: F = surface.border;
        let ind_x_border: usize = ((x_border - x.x0[0]) / x.dx[0]).round() as usize;
        let mut dpsi_dx = psi.psi.clone();
        gauss::fft_dpsi_dx(&mut dpsi_dx, &x, &p);
        let instance_flow: C = (1..(x.n[1] - 1))
            .into_par_iter()
            .map(|i| {
                // let deriv_x = (psi.psi[(ind_x_border + 1, i)] - psi.psi[(ind_x_border - 1, i)])
                //     / (2.0 * x.dx[0]);
                let val = psi.psi[(ind_x_border, i)];
                let xs = [x.grid[0][ind_x_border], x.grid[1][i]];
                let xs_pdx = [x.grid[0][ind_x_border + 1], x.grid[1][i]];
                let xs_mdx = [x.grid[0][ind_x_border - 1], x.grid[1][i]];
                let deriv_x = dpsi_dx[(ind_x_border, i)];
                // (gauss.wf(xs_pdx, t.current) - gauss.wf(xs_mdx, t.current)) / (2.0 * x.dx[0]);
                // let deriv_x = gauss.dwf_dx(xs, t.current);
                // let val = gauss.wf(xs, t.current);
                I / 2.0 * (val * deriv_x.conj() - val.conj() * deriv_x)
            })
            .sum::<C>()
            * x.dx[1];
        examine_flow.instance_flow.push(instance_flow);
        examine_flow.time_instance.push(t.current);
        //============================================================
        //                Analit Flow Square
        //============================================================
        let instance_analit_flow: C = (1..(x.n[1] - 1))
            .into_par_iter()
            .map(|i| {
                let x_point = x.grid[0][ind_x_border];
                let y_point = x.grid[1][i];
                gauss.j([x_point, y_point], t.current)[0]
            })
            .sum::<C>()
            * x.dx[1];
        analit_flow.instance_flow.push(instance_analit_flow);
        analit_flow.time_instance.push(t.current);

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
    let examine_total_flow = examine_flow.compute_total_flow(t.t_step());
    let analit_total_flow = analit_flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!("examine_total_flow = {}", examine_total_flow);
    println!("analit_total_flow = {}", analit_total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
    // график потока
    flow.plot_flow("src/tests/out/flow/flow_graph.png");
    examine_flow.plot_flow("src/tests/out/flow/examine_flow_graph.png");
    analit_flow.plot_flow("src/tests/out/flow/analit_flow_graph.png");
}
