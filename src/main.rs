#[macro_use]
extern crate fstrings;

mod evolution;
mod field;
mod flow;
mod gauge;
mod heatmap;
mod parameters;
mod potentials;
mod tsurff;
mod volkov;
mod wave_function;
use evolution::{EvolutionSSFM, SSFM};
use field::Field2D;
use flow::*;
use gauge::{LenthGauge, VelocityGauge};
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use parameters::{Pspace, Tspace, Xspace};
use potentials::AtomicPotential;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::process::exit;
use std::time::Instant;
use tsurff::*;
use wave_function::WaveFunction;

type F = f32;
type C = Complex<f32>;

fn main() {
    catalogs_check();
    // пути к сохраненным массивам
    let x_dir_path = "arrays_saved";
    let psi_path = "arrays_saved/psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 100, 30);
    t.save_grid("arrays_saved/time_evol/t.npy").unwrap();

    // задаем координатную сетку
    let x = Xspace::load(x_dir_path, 2);

    // инициализируем импульсную сетку на основе координатной сетки
    let p = Pspace::init(&x);
    p.save("arrays_saved/").unwrap();

    // инициализируем внешнее поле
    let field = Field2D {
        amplitude: 0.038,
        omega: 0.04,
        N: 3.0,
        x_envelop: 30.0001,
    };
    // указываем калибровку поля
    let gauge = LenthGauge::new(&field);

    // инициализируем структуру для SSFM эволюции (решатель ур. Шредингера)
    let mut ssfm = SSFM::new(&gauge, &x, &p);

    // t-surff
    // let surf = Circle::new(25.0, 50);
    // let mut tsurff = Tsurff::new(gauge, surf, &x, &p);

    // создаем структуру для потока вероятности через поверхность
    let surface = Circle::new(25.0, 50);
    let mut flow = Flow::new(&gauge, &surface);

    // генерация начальной волновой функции psi
    let mut psi = WaveFunction::init_from_file(psi_path, &x);
    // нормируем на всякий случай
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
        let time_ssfm = Instant::now();
        ssfm.time_step_evol(
            &mut psi,
            &mut t,
            Some(f!("arrays_saved/time_evol/psi_x_lg/psi_x_t_{i}.npy").as_str()),
            None, // Some(f!("arrays_saved/time_evol/psi_x/psi_x_t_{i}.npy").as_str()),
                  // Some(f!("arrays_saved/time_evol/psi_p/psi_p_t_{i}.npy").as_str()),
        );
        println!("SSFM = {:.3}", time_ssfm.elapsed().as_secs_f32());
        //============================================================
        //                       t-SURFF
        //============================================================
        // let tsurff_time = Instant::now();
        // tsurff.time_integration_step(&psi, &t);
        // println!("t-surff = {:.3}", tsurff_time.elapsed().as_secs_f32());
        //============================================================
        //                       Flow
        //============================================================
        let flow_time = Instant::now();
        flow.add_instance_flow(&psi, t.current);
        println!("flow_time = {}", flow_time.elapsed().as_secs_f32());

        //============================================================
        println!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )
    }
    flow.plot_flow("flow_graph.png");
    let total_flow = flow.compute_total_flow(t.t_step());
    println!("total_flow = {}", total_flow);
    println!(
        "total_flow + prob_in_box = {}",
        total_flow.re + psi.prob_in_numerical_box()
    );
}

fn save1dc(arr: Array1<C>, path: &str) {
    let writer = BufWriter::new(File::create(path).unwrap());
    arr.write_npy(writer).unwrap();
}

fn save2dc(arr: Array2<C>, path: &str) {
    let writer = BufWriter::new(File::create(path).unwrap());
    arr.write_npy(writer).unwrap();
}

fn catalogs_check() {
    use std::fs;
    use std::path::Path;
    let paths = [
        "arrays_saved",
        "arrays_saved/time_evol",
        "arrays_saved/time_evol/psi_x",
        "arrays_saved/time_evol/psi_p",
    ];
    for path in paths {
        if !Path::new(path).exists() {
            fs::create_dir(path).unwrap();
        }
    }
}
