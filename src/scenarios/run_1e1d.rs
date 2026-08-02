#![allow(dead_code, non_snake_case, unused_variables, unused_imports)]

//! Параметризованный ОДНОЭЛЕКТРОННЫЙ расчёт 1e1d (внутренний/внешний электрон —
//! различие только в конфиге: потенциал и начальная волна). Запуск:
//! cargo run --bin run_1e1d -- --config path/to/params.toml

mod config;
mod log;
mod manifest;
mod potential;

use config::Config;
use rssfm::common::{particle::Particle, tspace::Tspace};
use rssfm::config::{F, PI};
use rssfm::dim1::{
    field::UnipolarPulse1D,
    gauge::LenthGauge1D,
    ioniz_prob::IonizProb1D,
    space::Xspace1D,
    ssfm::SSFM1D,
    time_fft::TimeFFT,
    wave_function::WaveFunction1D,
};
use rssfm::traits::{
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use rssfm::utils::plot_log::plot_log;
use std::time::Instant;

const AU_TO_FS: F = 2.418_884_3e-2;

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

    let out = cfg.output().directory.clone();
    let tp = cfg.resolve_time();
    let mut t = Tspace::new(tp.t0 as F, tp.dt as F, tp.n_steps, tp.nt);
    if cfg.output().save_time_series_grid {
        t.save_grid(format!("{out}/time_evol/t.npy").as_str()).unwrap();
    }

    let mut psi = WaveFunction1D::init_from_hdf5(&cfg.initial_root());

    let g = cfg.resolve_grid();
    let x = Xspace1D::new([g.x0[0] as F], [g.dx[0] as F], [g.n[0]]);
    if cfg.output().save_time_series_grid {
        x.save_as_npy(format!("{out}/time_evol").as_str()).unwrap();
    }
    let (do_extend, do_norm) = cfg.init_flags();
    if do_extend {
        psi.extend(&x);
    }
    if do_norm {
        psi.normalization_by_1();
    }

    let ap = cfg.resolve_parameters();
    let atomic_potential = potential::atomic_1d(cfg.atomic_model(), ap);
    let ab = cfg.resolve_abs();
    let absorbing_potential = potential::absorbing_1d(ab.model, ab.alpha as F, ab.region, ab.radius);

    let fr = cfg.resolve_field();
    let field = field1d(&fr.pulse_shape, fr.amplitude as F, fr.omega as F, fr.x_envelop as F);
    let gauge = LenthGauge1D::new(&field);

    let particles: Vec<Particle> = cfg
        .resolve_particles()
        .iter()
        .map(|(dim, mass, charge)| Particle { dim: *dim, mass: *mass as F, charge: *charge as F })
        .collect();

    let mut ssfm = SSFM1D::new(&particles, &gauge, &psi.x, atomic_potential, absorbing_potential);

    // (опционально) временное Фурье-преобразование — для расчёта основного состояния
    let mut time_fft = cfg.time_fft().map(|tf| {
        let tf_fft = TimeFFT::new(t.clone(), [tf.point[0] as F], &psi.x);
        (tf, tf_fft)
    });

    // (опционально) вероятность ионизации IonizProb1D по поверхностям x_surf
    let mut ioniz_prob = cfg
        .resolve_x_surf()
        .map(|xs| IonizProb1D::new(xs.iter().map(|&v| v as F).collect(), t.get_grid()));

    let total_time = Instant::now();
    for i in 0..t.nt {
        let time_step = Instant::now();
        scenario_log!(
            &format!("{out}/output.log"),
            "STEP {}/{}, t.current={:.5}, norm = {}, prob_in_box = {}",
            i, t.nt, t.current, psi.norm(), psi.prob_in_numerical_box()
        );
        scenario_measure_time!(&format!("{out}/output.log"), "SSFM", {
            ssfm.time_step_evol(
                &mut psi,
                &mut t,
                Some(&mut |psi, t| momentum_processing(&cfg, psi, t, i)),
            );
        });
        position_processing(&cfg, &psi, &t, i);
        if let Some(ip) = &mut ioniz_prob {
            ip.add(&psi);
        }
        if let Some((_, tf_fft)) = time_fft.as_mut() {
            tf_fft.add_psi_in_point(&psi);
        }
        scenario_log!(
            &format!("{out}/output.log"),
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )
    }

    // (опционально) вероятность ионизации — график и данные
    if let Some(ip) = ioniz_prob {
        ip.plot(format!("{out}/ioniz_prob.png").as_str());
        ip.save_as_hdf5(format!("{out}/ioniz_prob.hdf5").as_str());
    }

    // (опционально) спектр по окончании эволюции
    if let Some((tf, mut tf_fft)) = time_fft {
        tf_fft.compute_spectrum();
        if tf.save_data {
            tf_fft.save_as_hdf5(format!("{out}/time_fft.hdf5").as_str());
        }
        if tf.save_plot {
            let limits = tf.energy_limits.unwrap_or_else(|| vec![-40.0, 40.0]);
            tf_fft.plot_log(format!("{out}/time_fft.png").as_str(), [limits[0] as F, limits[1] as F]);
        }
    }

    manifest::write_manifest(&cfg, &out);
}

fn momentum_processing(cfg: &Config, psi: &WaveFunction1D, _t: &Tspace, i_step: usize) {
    let out = &cfg.output().directory;
    if cfg.output().save_wavefunctions_hdf5 {
        psi.save_as_hdf5(format!("{out}/time_evol/psi_p/psi_p_{i_step}.hdf5").as_str());
    }
    if cfg.output().save_plots_png {
        psi.plot(format!("{out}/imgs/time_evol/psi_p/psi_p_{i_step}.png").as_str(), [-0.8, 0.8]);
    }
}

fn position_processing(cfg: &Config, psi: &WaveFunction1D, t: &Tspace, i_step: usize) {
    let out = &cfg.output().directory;
    if cfg.output().save_wavefunctions_hdf5 {
        psi.save_as_hdf5(format!("{out}/time_evol/psi_x/psi_x_{i_step}.hdf5").as_str());
    }
    if cfg.output().save_plots_png {
        psi.plot(format!("{out}/imgs/time_evol/psi_x/psi_x_{i_step}.png").as_str(), [-0.1, 0.8]);
    }
}

fn field1d(pulse_shape: &str, amplitude: F, omega: F, x_envelop: F) -> UnipolarPulse1D {
    match pulse_shape {
        "UnipolarPulse1D" => UnipolarPulse1D::new(amplitude, omega, x_envelop),
        other => panic!("неизвестная форма поля 1d: {other}"),
    }
}