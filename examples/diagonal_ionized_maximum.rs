use ndarray::prelude::*;
use rssfm::config::F;
use rssfm::dim4::wave_function::WaveFunction4D;
use rssfm::dim4::wave_function_processing::probability_density_2d::ProbabilityDensity2D;
use rssfm::measure_time;
use rssfm::traits::wave_function::WaveFunction;
use rssfm::utils::diagonal::Diagonal;
use rssfm::utils::processing::create_zeroed_wavefunction;

fn main() {
    for i in 0..45 {
        println!("STEP i={:?}", i);
        let prob_dens = ProbabilityDensity2D::init_from_hdf5(format!("/home/denis/akula_home/br2e2d_E0035_T2_FFT_MD_int_new/RSSFM2D/src/out/time_evol/psi_x/prob_dense_cut5/x1x2_{i}.hdf5").as_str());
        let diagonal = Diagonal::init_from(&prob_dens.probability_density, &prob_dens.axes[0]);
        let local_maxima = diagonal.get_local_maxima_above(1e-8);
        for k in 0..local_maxima.0.len() {
            if local_maxima.2[k] > 5.0 {
                println!(
                    "{:?}, {:?}, {:?}",
                    local_maxima.0[k], local_maxima.1[k], local_maxima.2[k]
                );
            }
        }
    }
}
