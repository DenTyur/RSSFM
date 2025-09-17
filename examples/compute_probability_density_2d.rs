use rssfm::dim4::probability_density_2d::ProbabilityDensity2D;
use rssfm::dim4::wave_function::WaveFunction4D;
use rssfm::traits::wave_function::WaveFunction;

fn main() {
    println!("Запуск probability_density_2d...");

    let psi = WaveFunction4D::init_from_hdf5(
        "/home/denis/Programs/atoms_and_ions/DATA/br/br2e2d_N64_dx05_interact.hdf5",
    );
    psi.plot_slice_log(
        "./slice.png",
        [1e-8, 1e-6],
        [None, Some(0.0), None, Some(0.0)],
    );

    let prob_density = ProbabilityDensity2D::compute_from_wf4d(&psi, [0, 2], [1, 3], None);
    prob_density.plot_log("./prob_density.png", [1e-8, 1e-6]);
    prob_density.save_as_hdf5("./prob_density.hdf5");
    println!("Завершен успешно!");
}
