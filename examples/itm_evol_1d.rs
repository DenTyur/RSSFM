use rayon::prelude::*;
use rssfm::common::particle::Particle;
use rssfm::common::tspace::Tspace;
use rssfm::common::units::AU_TO_EV;
use rssfm::config::{C, F};
use rssfm::dim1::{
    fft_maker::FftMaker1D, field::UnipolarPulse1D, gauge::LenthGauge1D, space::Xspace1D,
    ssfm::SSFM1D, ssfm_in_imaginary_time::SSFM1DinInaginaryTime, time_fft::TimeFFT,
    wave_function::WaveFunction1D,
};
use rssfm::measure_time;
use rssfm::potentials::absorbing_potentials::absorbing_potential_1d;
use rssfm::potentials::potentials;
use rssfm::print_and_log;
use rssfm::traits::{
    fft_maker::FftMaker,
    field::Field,
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    ssfm_in_imaginary_time::SSFMinInaginaryTime,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use std::time::Instant;

fn main() {
    // префикс для сохранения
    let out_prefix = ".";
    // пути к сохраненным массивам
    // let x_dir_path = ".";
    // let psi_path = "psi_initial.npy";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.01, 20, 500);
    let save_step: usize = 100;
    t.save_grid(format!("{out_prefix}/time_evol/t.npy").as_str())
        .unwrap();

    // задаем координатную сетку и в.ф. из .npy
    let x = Xspace1D::new([-30.0], [0.5], [120]);
    let mut psi = WaveFunction1D::init_oscillator_1d(x);

    psi.normalization_by_1();
    psi.plot(format!("{out_prefix}/psi_init.png").as_str(), [-0.1, 1.0]);

    // инициализируем внешнее поле
    let field = UnipolarPulse1D {
        amplitude: 0.0,
        omega: 0.04,
        x_envelop: 30.0001,
    };

    // указываем калибровку поля
    let gauge = LenthGauge1D::new(&field);

    // инициализируем структуру для SSFM эволюции (решатель ур. Шредингера)
    let abs_pot = |x: [F; 1]| absorbing_potential_1d(x, 50.0, 0.4);
    let atomic_pot = |x: [F; 1]| potentials::br_1e1d_inner(x);

    let electron1 = Particle {
        dim: 1,
        mass: 1.0,
        charge: -1.0,
    };
    let particles = [electron1];
    let mut ssfm_in_imaginary_time =
        SSFM1DinInaginaryTime::new(&particles, &gauge, &psi.x, atomic_pot, abs_pot);

    // эволюция
    let mut energy: F = compute_energy(&mut psi, atomic_pot).re * AU_TO_EV;
    println!("energy_init = {:?}", energy);
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
            ssfm_in_imaginary_time.time_step_evol(&mut psi, &mut t, None);
            psi.normalization_by_1();
        });
        let energy_current: F = compute_energy(&mut psi, atomic_pot).re * AU_TO_EV;
        let d_energy: F = (energy_current - energy).abs();
        print_and_log!("{:?}\t{:?}\t{:?}", energy, energy_current, d_energy);
        energy = energy_current;
        if d_energy < 1e-5 {
            break;
        }
    }
    // Сохраняем финальную в.ф. в psi_initial
    // psi.normalization_by_1();
    // psi.save_as_npy("psi_initial.npy");
    // psi.x.save_as_npy(".");
}

fn compute_energy(wf: &mut WaveFunction1D, atomic_pot: fn([F; 1]) -> F) -> C {
    compute_kinetic_energy(wf) + compute_potential_energy(wf, atomic_pot)
}

fn compute_potential_energy(wf: &mut WaveFunction1D, atomic_pot: fn([F; 1]) -> F) -> C {
    let mut potential_energy = C::new(0.0, 0.0);
    wf.psi
        .iter()
        .zip(wf.x.grid[0].iter())
        .for_each(|(psi, x)| potential_energy += psi.norm_sqr() * atomic_pot([*x]));
    potential_energy * wf.x.dx[0]
}

fn compute_kinetic_energy(wf: &mut WaveFunction1D) -> C {
    let mut fft_maker = FftMaker1D::new(&wf.x.n);
    // переходим в импульсное представление
    fft_maker.modify_psi(wf);
    fft_maker.do_fft(wf);
    // вычисляем кинетическую энергию
    let mut kinetic_energy = C::new(0.0, 0.0);
    wf.psi.iter().zip(wf.p.grid[0].iter()).for_each(|(psi, p)| {
        kinetic_energy += psi.norm_sqr() * p * p / 2.0;
    });
    // возвращаем волновую функцию в координатное представление
    fft_maker.do_ifft(wf);
    fft_maker.demodify_psi(wf);

    kinetic_energy * wf.p.dp[0]
}
