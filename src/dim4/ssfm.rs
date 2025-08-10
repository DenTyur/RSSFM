use super::fft_maker::FftMaker4D;
use super::space::Xspace4D;
use super::wave_function::WaveFunction4D;
use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm::{GaugedEvolutionSSFM, SSFM};
use crate::traits::wave_function::WaveFunction;

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
        psi_p_save_path: Option<(&str, &str, [F; 2])>,
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
            wf.save_as_npy(path.0).unwrap();
        }
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
