mod common;
mod config;
mod dim2;
mod dim4;
mod macros;
mod potentials;
mod traits;
mod utils;
use crate::common::tspace::Tspace;
use crate::config::F;
use crate::dim2::{
    field::Field2D,
    flow::{Flow2D, Square},
    gauge::VelocityGauge2D,
    space::{Pspace2D, Xspace2D},
    ssfm::SSFM2D,
    tsurff::Tsurff2D,
    wave_function::WaveFunction2D,
};
use crate::potentials::{absorbing_potentials::absorbing_potential, potentials::br_1e2d_external};
use crate::traits::{space::Space, ssfm::SSFM, tsurff::Tsurff, wave_function::WaveFunction};
use std::time::Instant;

fn main() {
    // префикс для сохранения
    let out_prefix = "out/dim2";
    // пути к сохраненным массивам
    let x_dir_path = "arrays_saved";
    let psi_path = "arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 100, 25);
    t.save_grid(format!("{out_prefix}/time_evol/t.npy").as_str())
        .unwrap();

    // задаем координатную сетку
    let x = Xspace2D::load_from_npy(x_dir_path);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace2D::init(&x);
    p.save_as_npy("arrays_saved/").unwrap();

    // инициализируем внешнее поле
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    // указываем калибровку поля
    let gauge = VelocityGauge2D::new(&field);

    // инициализируем структуру для SSFM эволюции (решатель ур. Шредингера)
    let abs_pot = |x: [F; 2]| absorbing_potential(x, 30.0, 0.02);
    let mut ssfm = SSFM2D::new(&gauge, &x, br_1e2d_external, abs_pot);

    // создаем структуру для потока вероятности через поверхность
    let surface = Square::new(20.0, &x);
    let mut flow = Flow2D::new(&gauge, &surface);

    // t-surff
    let mut tsurff = Tsurff2D::new(&gauge, &surface, &p, Some(3.0));

    // генерация начальной волновой функции psi
    let mut psi = WaveFunction2D::init_from_npy(psi_path, x.clone());
    psi.normalization_by_1();

    // эволюция
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
        //                       SSFM
        //============================================================
        measure_time!("SSFM", {
            ssfm.time_step_evol(
                &mut psi, &mut t,
                None,
                // Some(f!("arrays_saved/time_evol/psi_x/psi_x_t_{i}.npy").as_str()),
                // Some(f!("arrays_saved/time_evol/psi_p/psi_p_t_{i}.npy").as_str()),
            );
        });
        // график волновой функции
        // if i % 100 == 0 {
        psi.plot_log(
            format!("{out_prefix}/imgs/time_evol/psi_x/psi_x_t_{i}.png").as_str(),
            [1e-8, 1.0],
        );
        // }
        //обновление производных
        measure_time!("update_deriv", {
            psi.update_derivatives();
        });
        //============================================================
        //                       t-SURFF
        //============================================================
        measure_time!("tsurff", {
            tsurff.time_integration_step(&psi, &t);
            if i % 100 == 0 {
                tsurff.plot_log(
                    format!("{out_prefix}/tsurff/tsurff{i}.png").as_str(),
                    [1e-5, 1.0],
                );
            }
        });
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
        )
    }

    flow.plot_flow(format!("{out_prefix}/imgs/flow.png").as_str());
    let total_flow = flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
}
