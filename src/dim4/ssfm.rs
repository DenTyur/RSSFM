use super::fft_maker::FftMaker4D;
use super::space::Xspace4D;
use super::wave_function::WaveFunction4D;
use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm::{GaugedEvolutionSSFM, SSFM};
use crate::traits::wave_function::WaveFunction;
// use crate::utils::integrate_y1y2::integrate_y1y2;
// use ndarray::prelude::*;
// use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
// use num_complex::Complex;
// use rayon::prelude::*;
// use std::fs::File;
// use std::io::BufWriter;

pub struct SSFM4D<'a, G>
where
    G: GaugedEvolutionSSFM<4, WF = WaveFunction4D>,
{
    particles: &'a [Particle],
    potential: fn([F; 4]) -> F,
    absorbing_potential: fn([F; 4]) -> C,
    gauge: &'a G,
    fft_maker: FftMaker4D,
}

impl<'a, G> SSFM4D<'a, G>
where
    G: GaugedEvolutionSSFM<4, WF = WaveFunction4D>,
{
    pub fn new(
        particles: &'a [Particle],
        gauge: &'a G,
        x: &Xspace4D,
        potential: fn([F; 4]) -> F,
        absorbing_potential: fn([F; 4]) -> C,
    ) -> Self {
        let fft_maker = FftMaker4D::new(&x.n);
        Self {
            particles,
            gauge,
            fft_maker,
            potential,
            absorbing_potential,
        }
    }
}

/// Реализация эволюции на временной шаг методом SSFM
impl<'a, G> SSFM for SSFM4D<'a, G>
where
    G: GaugedEvolutionSSFM<4, WF = WaveFunction4D>,
{
    type WF = WaveFunction4D;

    fn time_step_evol(
        &mut self,
        wf: &mut WaveFunction4D,
        t: &mut Tspace,
        psi_p_save_path: Option<(&str, isize, &str, [F; 2])>,
    ) {
        self.fft_maker.modify_psi(wf);
        self.gauge.x_evol_half(
            self.particles,
            wf,
            t.current,
            t.dt,
            self.potential,
            self.absorbing_potential,
        );

        for _i in 0..t.n_steps - 1 {
            self.fft_maker.do_fft(wf);
            // Можно оптимизировать p_evol
            self.gauge.p_evol(self.particles, wf, t.current, t.dt);
            self.fft_maker.do_ifft(wf);
            self.gauge.x_evol(
                self.particles,
                wf,
                t.current,
                t.dt,
                self.potential,
                self.absorbing_potential,
            );
            t.current += t.dt;
        }

        self.fft_maker.do_fft(wf);
        self.gauge.p_evol(self.particles, wf, t.current, t.dt);
        if let Some(path) = psi_p_save_path {
            wf.save_sparsed_as_npy(path.0, path.1).unwrap();
            wf.plot_momentum_slice_log(path.2, path.3, [None, Some(0.0_f32), None, Some(0.0_f32)]);
        }
        //==========================================
        // let psi_p_squared_px1px2: Array<F, Ix2> =
        //     integrate_y1y2(&wf.psi, &wf.p.grid[1], &wf.p.grid[3], 0.0);
        //
        // let path_sq = "out/psi_p_squared_px1px2"
        // check_path!(path);
        // let writer = BufWriter::new(File::create(path)?);
        // psi_p_squared_px1px2.write_npy(writer)?;

        //==========================================
        self.fft_maker.do_ifft(wf);
        self.gauge.x_evol_half(
            self.particles,
            wf,
            t.current,
            t.dt,
            self.potential,
            self.absorbing_potential,
        );
        self.fft_maker.demodify_psi(wf);
        t.current += t.dt;
    }
}
