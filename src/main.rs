#![allow(dead_code, non_snake_case, unused_variables, unused_imports)]
use ndarray::prelude::*;
use rayon::prelude::*;
use rssfm::common::{particle::Particle, tspace::Tspace};
use rssfm::config::{C, F, PI};
use rssfm::dim2::{
    field::UnipolarPulse1e2d,
    gauge::{LenthGauge2D, VelocityGauge2D},
};
use rssfm::dim4::{
    fft_maker::FftMaker4D,
    gauge::{LenthGauge4D, VelocityGauge4D},
    probability_density_2d::ProbabilityDensity2D,
    space::Xspace4D,
    ssfm::SSFM4D,
    time_fft::TimeFFT,
    wave_function::WaveFunction4D,
};
use rssfm::measure_time;
use rssfm::potentials::absorbing_potentials::{
    absorbing_potential_4d, absorbing_potential_4d_asim,
};
use rssfm::potentials::potentials;
use rssfm::print_and_log;
use rssfm::traits::fft_maker::FftMaker;
use rssfm::traits::{
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use rssfm::utils::plot_log::plot_log;
use std::array;
use std::time::Instant;

fn main() {
    // префикс для сохранения
    let out_prefix = "./out";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 10, 45);
    t.save_grid(format!("{out_prefix}/time_evol/t.npy").as_str())
        .unwrap();

    // начальная волновая функция
    let mut psi = WaveFunction4D::init_from_hdf5("br2e2d_N64_dx05_interact.hdf5");

    // расширяем сетку
    let x = Xspace4D::new([-20., -50., -20., -50.], [0.5; 4], [240, 200, 240, 200]);
    // сохраняем расширенную сетку
    x.save_as_npy(format!("{out_prefix}/time_evol").as_str())
        .unwrap();
    psi.extend(&x);
    psi.normalization_by_1();

    // Пользовательская обработка до эволюции
    initial_processing(&psi, &t, out_prefix);

    // атомный и поглощающий потенциалы
    let absorbing_potential = |x: [F; 4]| {
        absorbing_potential_4d_asim(
            x,
            [[-15., 500.], [-45., 45.], [-15., 500.], [-45., 45.]],
            0.4,
        )
    };
    let atomic_potential = |x: [F; 4]| potentials::br_2e2d(x);

    // инициализируем внешнее поле <-- сложно унифицировать в конфиг
    const AU_TO_FS: F = 2.418_884_3e-2;
    let T_fs: F = 2.; //fs
    let T_au: F = T_fs / AU_TO_FS;
    let omega: F = PI / T_au;
    let field = UnipolarPulse1e2d {
        amplitude: 0.035,
        omega,
        x_envelop: 50.0001,
    };
    // указываем калибровку поля
    let gauge = LenthGauge2D::new(&field);

    // задаем частицы: два двумерных электрона
    let electron1 = Particle {
        dim: 2,
        mass: 1.0,
        charge: -1.0,
    };
    let electron2 = electron1;
    let particles = [electron1, electron2];

    // инициализируем структуру для SSFM эволюции (решатель ур. Шредингера)
    let mut ssfm = SSFM4D::new(
        &particles,
        &gauge,
        &psi.x,
        atomic_potential,
        absorbing_potential,
    );

    // временное FFT
    let min_length_of_axes = psi.x.grid.iter().map(|axis| axis.len()).min().unwrap_or(0);
    let mut time_fft_arr: Vec<TimeFFT> = (0..min_length_of_axes)
        .map(|i| TimeFFT::new(t.clone(), [x.grid[0][i], 0.1, 0.1, 0.1], &x))
        .collect();

    // эволюция
    let total_time = Instant::now();
    for i in 0..t.nt {
        let time_step = Instant::now();
        print_and_log!(
            "STEP {}/{}, t.current={:.5}, norm = {}, prob_in_box = {}",
            i,
            t.nt,
            t.current,
            psi.norm(),
            psi.prob_in_numerical_box(),
        );
        //============================================================
        //              SSFM and processing in current time
        //============================================================
        measure_time!("SSFM", {
            ssfm.time_step_evol(
                &mut psi,
                &mut t,
                Some(&mut |psi, t| momentum_processing(psi, t, i, out_prefix)),
            );
        });
        position_processing(&psi, &t, i, out_prefix);
        // ============================================================
        //                    Временное FFT
        // ============================================================
        // добавляем элемент для временного fft
        for time_fft in time_fft_arr.iter_mut() {
            time_fft.add_psi_in_point(&psi);
        }
        //============================================================
        print_and_log!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )
    }

    //==========================================================================
    //=========================== Постобработка ================================
    //==========================================================================
    // ------------ FFT Интегрируем по всем точкам -----------------------------
    let mut energy_spectrum_total: Array1<F> = Array::zeros(t.nt);

    for time_fft in time_fft_arr.iter_mut() {
        time_fft.compute_spectrum();

        time_fft
            .psi_fft
            .iter()
            .zip(energy_spectrum_total.iter_mut())
            .par_bridge()
            .for_each(|(psi_elem, energy_spectrum_elem)| {
                *energy_spectrum_elem += psi_elem.norm_sqr();
            });
    }
    energy_spectrum_total *= psi.x.dx[0];

    plot_log(
        time_fft_arr[0].energy.clone(),
        energy_spectrum_total,
        "energy [eV]",
        format!("{out_prefix}/energy_spectrum_total.png").as_str(),
    );
    // ------------------------строим временное fft-----------------------------
    for i in [20, 60, 80, 100, 120, 140, 160, 180] {
        let xcurrent: F = psi.x.grid[0][i];
        time_fft_arr[i].compute_spectrum();
        time_fft_arr[i].plot_log(format!("{out_prefix}/time_fft_x_is{xcurrent}.png").as_str());
    }

    // строим временное fft
    // time_fft.compute_spectrum();
    // time_fft.plot_log("time_fft.png");
    // println!(
    //     "energy = {}, {}",
    //     time_fft.energy[[0]],
    //     time_fft.energy[[t.nt - 1]]
    // );
    // Сохраняем финальную в.ф. в psi_initial
    // psi.normalization_by_1();
    // psi.save_as_npy("psi_initial.npy");
    // psi.x.save_as_npy(".");
}

fn initial_processing(psi: &WaveFunction4D, t: &Tspace, out_prefix: &str) {
    psi.plot_slice_log(
        format!("{out_prefix}/psi_zero_slice_x0x1.png").as_str(),
        [1e-8, 1.0],
        [None, None, Some(0.0_f32), Some(0.0_f32)],
    );
    psi.plot_slice_log(
        format!("{out_prefix}/psi_zero_slice_x0x3.png").as_str(),
        [1e-8, 1.0],
        [None, Some(0.0_f32), Some(0.0_f32), None],
    );
    psi.plot_slice_log(
        format!("{out_prefix}/psi_zero_slice_x1x2.png").as_str(),
        [1e-8, 1.0],
        [Some(0.0_f32), None, None, Some(0.0_f32)],
    );
    psi.plot_slice_log(
        format!("{out_prefix}/psi_zero_slice_x0x2.png").as_str(),
        [1e-8, 1.0],
        [None, Some(0.0_f32), None, Some(0.0_f32)],
    );
    psi.plot_slice_log(
        format!("{out_prefix}/psi_zero_slice_x2x3.png").as_str(),
        [1e-8, 1.0],
        [Some(0.0_f32), Some(0.0_f32), None, None],
    );

    let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [0, 1], [2, 3], None);
    prob_dens.plot_log(
        format!("{out_prefix}/prob_dense_x0x1.png").as_str(),
        [1e-8, 1e-6],
    );

    let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [0, 3], [1, 2], None);
    prob_dens.plot_log(
        format!("{out_prefix}/prob_dense_x0x3.png").as_str(),
        [1e-8, 1e-6],
    );

    let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [1, 2], [0, 3], None);
    prob_dens.plot_log(
        format!("{out_prefix}/prob_dense_x1x2.png").as_str(),
        [1e-8, 1e-6],
    );

    let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [0, 2], [1, 3], None);
    prob_dens.plot_log(
        format!("{out_prefix}/prob_dense_x0x2.png").as_str(),
        [1e-8, 1e-6],
    );

    let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [2, 3], [0, 1], None);
    prob_dens.plot_log(
        format!("{out_prefix}/prob_dense_x2x3.png").as_str(),
        [1e-8, 1e-6],
    );
}

fn momentum_processing(psi: &WaveFunction4D, t: &Tspace, i_step: usize, out_prefix: &str) {
    let save_step: usize = 1;
    if save_step == 1 || i_step % save_step == 0 {
        // график среза py1=py2=0
        psi.plot_slice_log(
            format!("{out_prefix}/imgs/time_evol/psi_p/psi_p_t_{i_step}.png").as_str(),
            [1e-8, 1e-6],
            [None, Some(0.0_f32), None, Some(0.0_f32)],
        );
        // сохранение волновой функции в импульсном представлении
        // psi.save_sparsed_as_npy(
        //     format!("{out_prefix}/time_evol/psi_x/psi_x_t_{i_step}.npy").as_str(),
        //     4,
        // )
        // .unwrap();

        // интегрирование по py1py2 с разным вырезом серединки
        let cuts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for cut in cuts {
            let compute_time = Instant::now();
            let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [0, 2], [1, 3], Some(cut));
            print_and_log!(
                "prob_dens_momentum -- compute = {:.3}",
                compute_time.elapsed().as_secs_f32()
            );
            measure_time!("prob_dens_momentum -- plot = ", {
                prob_dens.plot_log(
                    format!(
                        "{out_prefix}/imgs/time_evol/psi_p/prob_dense_cut{cut}/px1px2_{i_step}.png"
                    )
                    .as_str(),
                    [1e-8, 1e-6],
                );
            });
            measure_time!("prob_dens_momentum -- save = ", {
                prob_dens.save_as_hdf5(
                    format!(
                        "{out_prefix}/time_evol/psi_p/prob_dense_cut{cut}/px1px2_{i_step}.hdf5"
                    )
                    .as_str(),
                );
            });
        }
    }
}

fn position_processing(psi: &WaveFunction4D, t: &Tspace, i_step: usize, out_prefix: &str) {
    let save_step: usize = 1;
    if save_step == 1 || i_step % save_step == 0 {
        // график среза волновой функции
        psi.plot_slice_log(
            format!("{out_prefix}/imgs/time_evol/psi_x/psi_x_t_{i_step}.png").as_str(),
            [1e-8, 1e-6],
            [None, Some(0.0_f32), None, Some(0.0_f32)],
        );
        // интегрирование по y1y2 с разным вырезом серединки
        let cuts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for cut in cuts {
            let compute_time = Instant::now();
            let prob_dens = ProbabilityDensity2D::compute_from_wf4d(psi, [0, 2], [1, 3], Some(cut));
            print_and_log!(
                "prob_dens_position -- compute = {:.3}",
                compute_time.elapsed().as_secs_f32()
            );
            measure_time!("prob_dens_position -- plot = ", {
                prob_dens.plot_log(
                    format!(
                        "{out_prefix}/imgs/time_evol/psi_x/prob_dense_cut{cut}/x1x2_{i_step}.png"
                    )
                    .as_str(),
                    [1e-8, 1e-6],
                );
            });
            measure_time!("prob_dens_position -- save = ", {
                prob_dens.save_as_hdf5(
                    format!("{out_prefix}/time_evol/psi_x/prob_dense_cut{cut}/x1x2_{i_step}.hdf5")
                        .as_str(),
                );
            });
        }
        // =================================================================================
        //              Импульсное распределение без центра r1 и r2 > 5.0
        // =================================================================================
        // Получим импульсное распределение однократной и двойной ионизации
        // 1. Вырезать серединку (заполнить ее нулями)
        let threshold: F = 5.0;
        let mut psi_zeroed = create_zeroed_wavefunction(psi, threshold);
        // 2. Сохранить обрезку
        let cuts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for cut in cuts {
            let compute_time = Instant::now();
            let prob_dens =
                ProbabilityDensity2D::compute_from_wf4d(&psi_zeroed, [0, 2], [1, 3], Some(cut));
            print_and_log!(
                "prob_dens_position -- compute = {:.3}",
                compute_time.elapsed().as_secs_f32()
            );
            measure_time!("prob_dens_position -- plot = ", {
                prob_dens.plot_log(
                    format!(
                        "{out_prefix}/imgs/time_evol/psi_zeroed_x/prob_dense_cut{cut}/x1x2_{i_step}.png"
                    )
                    .as_str(),
                    [1e-8, 1e-6],
                );
            });
            measure_time!("prob_dens_position -- save = ", {
                prob_dens.save_as_hdf5(
                    format!("{out_prefix}/time_evol/psi_zeroed_x/prob_dense_cut{cut}/x1x2_{i_step}.hdf5")
                        .as_str(),
                );
            });
        }
        // 3. Сделать прямое преобразование фурье
        let mut fft_maker = FftMaker4D::new(&psi_zeroed.x.n);
        fft_maker.modify_psi(&mut psi_zeroed);
        fft_maker.do_fft(&mut psi_zeroed);
        // 4. Сохранить в импульсном представлении обрезку
        let cuts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for cut in cuts {
            let compute_time = Instant::now();
            let prob_dens =
                ProbabilityDensity2D::compute_from_wf4d(&psi_zeroed, [0, 2], [1, 3], Some(cut));
            print_and_log!(
                "prob_dens_momentum -- compute = {:.3}",
                compute_time.elapsed().as_secs_f32()
            );
            measure_time!("prob_dens_momentum -- plot = ", {
                prob_dens.plot_log(
                    format!(
                        "{out_prefix}/imgs/time_evol/psi_zeroed_p/prob_dense_cut{cut}/px1px2_{i_step}.png"
                    )
                    .as_str(),
                    [1e-8, 1e-6],
                );
            });
            measure_time!("prob_dens_momentum -- save = ", {
                prob_dens.save_as_hdf5(
                    format!(
                        "{out_prefix}/time_evol/psi_zeroed_p/prob_dense_cut{cut}/px1px2_{i_step}.hdf5"
                    )
                    .as_str(),
                );
            });
        }

        // Срезы волновой функции y1y2 при x1=x2=local_max
    }
}

pub fn create_zeroed_wavefunction(original: &WaveFunction4D, threshold: F) -> WaveFunction4D {
    // Полная копия
    let mut new_wf = WaveFunction4D {
        psi: original.psi.clone(),
        dpsi_d0: original.dpsi_d0.clone(),
        dpsi_d1: original.dpsi_d1.clone(),
        dpsi_d2: original.dpsi_d2.clone(),
        dpsi_d3: original.dpsi_d3.clone(),
        x: original.x.clone(),
        p: original.p.clone(),
        representation: original.representation,
    };

    // Находим граничные индексы для каждой оси
    let idx_0 = find_boundary_index(&original.x.grid[0], threshold);
    let idx_1 = find_boundary_index(&original.x.grid[1], threshold);
    let idx_2 = find_boundary_index(&original.x.grid[2], threshold);
    let idx_3 = find_boundary_index(&original.x.grid[3], threshold);

    // Зануляем только нужную область
    zero_region(&mut new_wf.psi, idx_0, idx_1, idx_2, idx_3);

    new_wf
}

// Находит индекс, до которого нужно занулять
fn find_boundary_index(grid: &Array1<F>, threshold: F) -> usize {
    grid.iter()
        .position(|&x| x >= threshold)
        .unwrap_or(grid.len())
}

fn zero_region(array: &mut Array4<C>, idx_0: usize, idx_1: usize, idx_2: usize, idx_3: usize) {
    for i in 0..idx_0 {
        for j in 0..idx_1 {
            for k in 0..idx_2 {
                for l in 0..idx_3 {
                    array[[i, j, k, l]] = C::new(0.0, 0.0);
                }
            }
        }
    }
}
