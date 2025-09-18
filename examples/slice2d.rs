use rssfm::dim4::space::Xspace4D;
use rssfm::dim4::wave_function::WaveFunction4D;
use rssfm::dim4::wave_function_processing::probability_density_2d::ProbabilityDensity2D;
use rssfm::dim4::wave_function_processing::wave_function_slice_2d::WFSlice2D;
use rssfm::measure_time;
use rssfm::traits::wave_function::WaveFunction;
use rssfm::utils::processing::create_zeroed_wavefunction;

fn main() {
    println!("Запуск zeroed_wf...");

    let mut psi = WaveFunction4D::init_from_hdf5(
        "/home/denis/Programs/atoms_and_ions/DATA/br/br2e2d_N64_dx05_interact.hdf5",
    );
    let x = Xspace4D::new([-15., -10., -15., -10.], [0.5; 4], [60, 40, 60, 40]);
    psi.extend(&x);

    let slice = WFSlice2D::init_from_wf4d(&psi, [Some(0.0), None, Some(0.0), None]);
    slice.plot_log("./out/slice.png", [1e-8, 1e-6]);
}
