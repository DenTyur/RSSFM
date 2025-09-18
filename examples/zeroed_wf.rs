use rssfm::dim4::probability_density_2d::ProbabilityDensity2D;
use rssfm::dim4::wave_function::WaveFunction4D;
use rssfm::measure_time;
use rssfm::traits::wave_function::WaveFunction;
use rssfm::utils::processing::create_zeroed_wavefunction;

fn main() {
    println!("Запуск zeroed_wf...");

    let psi = WaveFunction4D::init_from_hdf5(
        "/home/denis/Programs/atoms_and_ions/DATA/br/br2e2d_N64_dx05_interact.hdf5",
    );

    let psi_zeroed = create_zeroed_wavefunction(&psi, 5.0);
    psi_zeroed.plot_slice_log(
        "./out/zeroed_slice.png",
        [1e-8, 1e-6],
        [None, Some(0.0), None, Some(0.0)],
    );

    let zeroed_prob_density =
        ProbabilityDensity2D::compute_from_wf4d(&psi_zeroed, [0, 2], [1, 3], Some(0.0));
    zeroed_prob_density.plot_log("./out/zeroed_prob_density.png", [1e-8, 1e-6]);
}
