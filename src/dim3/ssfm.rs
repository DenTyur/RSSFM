use super::fft_maker::FftMaker3D;
use super::space::Xspace3D;
use super::wave_function::WaveFunction3D;
use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm::{GaugedEvolutionSSFM, SSFM};
use crate::traits::wave_function::WaveFunction;

pub struct SSFM3D<'a, G>
where
    G: GaugedEvolutionSSFM<3, WF = WaveFunction3D>,
{
    particles: &'a [Particle],
    potential: fn([F; 3]) -> F,
    absorbing_potential: fn([F; 3]) -> C,
    gauge: &'a G,
    fft_maker: FftMaker3D,
}

impl<'a, G> SSFM3D<'a, G>
where
    G: GaugedEvolutionSSFM<3, WF = WaveFunction3D>,
{
    pub fn new(
        particles: &'a [Particle],
        gauge: &'a G,
        x: &Xspace3D,
        potential: fn([F; 3]) -> F,
        absorbing_potential: fn([F; 3]) -> C,
    ) -> Self {
        let fft_maker = FftMaker3D::new(&x.n);
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
impl<'a, G> SSFM for SSFM3D<'a, G>
where
    G: GaugedEvolutionSSFM<3, WF = WaveFunction3D>,
{
    type WF = WaveFunction3D;

    fn time_step_evol(
        &mut self,
        wf: &mut WaveFunction3D,
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
            // график волновой функции
            wf.save_as_npy(path.0).unwrap();
            // wf.plot_log(path.1, path.2);
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
