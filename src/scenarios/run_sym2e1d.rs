#![allow(dead_code, non_snake_case, unused_variables, unused_imports)]

//! Параметризованный расчёт симметризованного невзаимодействующего 2e1d из
//! готовых волн внутреннего/внешнего электронов. Сетка берётся из .hdf5 волновых
//! функций (поле WaveFunction.x), отдельная inner_grid_root не нужна; проверяется,
//! что сетки inner и external совпадают. Запуск:
//! cargo run --bin run_sym2e1d -- --config path/to/params.toml

mod config;
mod log;
mod manifest;

use config::Config;
use ndarray::Array1;
use ndarray_npy::ReadNpyExt;
use rssfm::config::F;
use rssfm::dim1::space::Xspace1D;
use rssfm::dim1::wave_function::WaveFunction1D;
use rssfm::dim2::{
    ioniz_prob::DoubleIonizProb2e1d, space::Xspace2D, wave_function::WaveFunction2D,
};
use rssfm::traits::wave_function::{SymmetrizedProduct, WaveFunction};
use rssfm::utils::symmetrized_noninteracting_wf_2e1d::SymNonintWF2e1d;
use std::fs::File;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| panic!("не передан --config path/to/params.toml"));
    let cfg = Config::load(&config_path).expect("конфиг");
    if args.iter().any(|a| a == "--parse-only") {
        println!("{cfg:#?}");
        return;
    }
    let sym = cfg
        .sym_nonint
        .as_ref()
        .unwrap_or_else(|| panic!("для run_sym1_2e1d нужна секция [\"sym noninteracting state\"]"));
    let out = &cfg.output().directory;

    let psi1_root = &sym.inner_wavefunction_root;
    let psi2_root = &sym.external_wavefunction_root;

    // сетку берём из волновой функции (0-й срезу), проверяем совпадение inner/external
    let wf1_g = WaveFunction1D::init_from_hdf5(format!("{psi1_root}/psi_x_0.hdf5").as_str());
    let wf2_g = WaveFunction1D::init_from_hdf5(format!("{psi2_root}/psi_x_0.hdf5").as_str());
    check_grids_match(&wf1_g.x, &wf2_g.x);
    let x = &wf1_g.x;
    let x2d = Xspace2D::new(
        [x.x0[0], x.x0[0]],
        [x.dx[0], x.dx[0]],
        [x.n[0], x.n[0]],
    );

    let reader = File::open(&sym.time_grid_path).unwrap();
    let t = Array1::<F>::read_npy(reader).unwrap();

    let mut ioniz_prob_norm = DoubleIonizProb2e1d::new(r_surf(&cfg), t.clone());
    let mut ioniz_prob_non_norm = ioniz_prob_norm.clone();

    for i in 0..t.len() {
        println!("i=============={}", i);
        let wf1 = WaveFunction1D::init_from_hdf5(format!("{psi1_root}/psi_x_{i}.hdf5").as_str());
        let wf2 = WaveFunction1D::init_from_hdf5(format!("{psi2_root}/psi_x_{i}.hdf5").as_str());
        let mut wf2e = WaveFunction2D::new_symmetrized_product(&wf1, &wf2);

        ioniz_prob_non_norm.add(&wf2e);

        scenario_log!(&format!("{out}/output.log"), "norm_inner: {:?}", wf1.norm());
        scenario_log!(&format!("{out}/output.log"), "norm_external: {:?}", wf2.norm());

        wf2e.normalization_by_1();
        scenario_log!(
            &format!("{out}/output.log"),
            "norm: {:?}, prob_in_box: {:?}",
            wf2e.norm(),
            wf2e.prob_in_numerical_box()
        );

        ioniz_prob_norm.add(&wf2e);

        if cfg.output().save_wavefunctions_hdf5 {
            wf2e.save_as_hdf5(format!("{out}/time_evol/wf2e1d_sym_nonint/wf2e1d_{i}.hdf5").as_str());
        }
        if cfg.output().save_plots_png {
            wf2e.plot_log(format!("{out}/imgs/wf2e1d_sym_nonint/wf2e1d_{i}.png").as_str(), [1e-8, 1e-6]);
        }
    }
    ioniz_prob_norm.plot(format!("{out}/ioniz_prob_norm.png").as_str());
    ioniz_prob_norm.save_as_hdf5(format!("{out}/ioniz_prob_norm.hdf5").as_str());
    ioniz_prob_non_norm.plot(format!("{out}/ioniz_prob_non_norm.png").as_str());
    ioniz_prob_non_norm.save_as_hdf5(format!("{out}/ioniz_prob_non_norm.hdf5").as_str());

    manifest::write_manifest(&cfg, out);
}

fn r_surf(cfg: &Config) -> Vec<F> {
    cfg.resolve_r_surf().iter().map(|&v| v as F).collect()
}

fn check_grids_match(a: &Xspace1D, b: &Xspace1D) {
    let same = a.n[0] == b.n[0]
        && (a.x0[0] - b.x0[0]).abs() < 1e-6
        && (a.dx[0] - b.dx[0]).abs() < 1e-6;
    assert!(same, "сетки inner и external не совпадают: {a:?} vs {b:?}");
    println!("проверка сеток inner/external: совпадают (n={}, x0={}, dx={})", a.n[0], a.x0[0], a.dx[0]);
}