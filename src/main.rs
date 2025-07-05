use rssfm::common::tspace::Tspace;
use rssfm::config::F;
use rssfm::dim2::ioniz_prob::IonizProb2D;
use rssfm::dim2::{
    field::Field2D,
    gauge::{LenthGauge2D, VelocityGauge2D},
    space::Xspace2D,
    ssfm::SSFM2D,
    time_fft::TimeFFT,
    wave_function::WaveFunction2D,
};
use rssfm::measure_time;
use rssfm::potentials::absorbing_potentials::absorbing_potential_2d_asim;
use rssfm::potentials::potentials;
use rssfm::print_and_log;
use rssfm::traits::{
    flow::{Flux, SurfaceFlow},
    space::Space,
    ssfm::SSFM,
    tsurff::Tsurff,
    wave_function::WaveFunction,
};
use std::time::Instant;

fn main() {
    // префикс для сохранения
    let out_prefix = "./out";

    // задаем параметры временной сетки
    let mut t = Tspace::new(0., 0.2, 200, 40);
    let save_step: usize = 1;
    t.save_grid(format!("{out_prefix}/time_evol/t.npy").as_str())
        .unwrap();

    // генерация начальной волновой функции psi
    let psi_path = "/home/denis/disk_storage/DATA/br/br2e1d_N120_dx05_interact.hdf5";
    let mut psi = WaveFunction2D::init_from_hdf5(psi_path);
    // расширяем сетку
    let x = Xspace2D::new([-100.0, -100.0], [0.5, 0.5], [2000, 2000]);
    psi.extend(&x);
    psi.normalization_by_1();
    psi.plot_log(format!("{out_prefix}/psi_init.png").as_str(), [1e-8, 1.0]);
    psi.x.save_as_npy(".");

    // атомный и поглощающий потенциалы
    let abs_pot = |x: [F; 2]| absorbing_potential_2d_asim(x, [[-50.0, 850.0], [-50.0, 850.0]], 0.4);
    let atomic_pot = |x: [F; 2]| potentials::soft_coulomb_2e1d_interact(x, -1.0, 1.66, 2.6);

    // вероятность двойной ионизации
    let mut ioniz_prob = IonizProb2D::new(
        [8.0, 10.0, 20.0, 30.0, 50.0, 100.0],
        psi.x.clone(),
        t.grid.clone(),
    );

    // инициализируем внешнее поле
    let field = Field2D {
        amplitude: 0.035,
        omega: 0.0018849555921538759,
        N: 3.0,
        x_envelop: 1000.0001,
    };

    // указываем калибровку поля
    let gauge = LenthGauge2D::new(&field);

    // инициализируем структуру для SSFM эволюции (решатель ур. Шредингера)
    let mut ssfm = SSFM2D::new(&gauge, &psi.x, atomic_pot, abs_pot);

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
        //                       SSFM
        //============================================================
        measure_time!("SSFM", {
            ssfm.time_step_evol(&mut psi, &mut t, None);
        });
        // график волновой функции
        if save_step == 1 || i % save_step == 0 {
            psi.plot_log(
                format!("{out_prefix}/imgs/time_evol/psi_x/psi_x_t_{i}.png").as_str(),
                [1e-8, 1.0],
            );
            //сохранение в.ф.
            psi.save_as_npy(format!("{out_prefix}/time_evol/psi_x/psi_t_{i}.npy").as_str());
        }
        // Ioniz Prob
        ioniz_prob.add(&psi.psi);
        //============================================================
        print_and_log!(
            "time_step = {:.3}, total_time = {:.3}",
            time_step.elapsed().as_secs_f32(),
            total_time.elapsed().as_secs_f32()
        )
    }

    ioniz_prob.plot(&t.grid, "ioniz_prob.png");
    ioniz_prob.save_as_hdf5("ioniz_prob.hdf5");
}
