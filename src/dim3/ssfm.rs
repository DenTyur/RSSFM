use super::fft_maker::FftMaker3D;
use super::space::Xspace3D;
use super::wave_function::WaveFunction3D;
use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm::{GaugedEvolutionSSFM, SSFM};
use crate::traits::wave_function::WaveFunction;

pub struct SSFM3D<'a, G, AP, AB>
where
    AP: Fn([F; 3]) -> F + Send + Sync,
    AB: Fn([F; 3]) -> C + Send + Sync,
    G: GaugedEvolutionSSFM<3, AP, AB, WF = WaveFunction3D>,
{
    particles: &'a [Particle],
    potential: AP,           // Изменено с fn на AP
    absorbing_potential: AB, // Изменено с fn на AB
    gauge: &'a G,
    fft_maker: FftMaker3D,
}

impl<'a, G, AP, AB> SSFM3D<'a, G, AP, AB>
where
    AP: Fn([F; 3]) -> F + Send + Sync,
    AB: Fn([F; 3]) -> C + Send + Sync,
    G: GaugedEvolutionSSFM<3, AP, AB, WF = WaveFunction3D>,
{
    pub fn new(
        particles: &'a [Particle],
        gauge: &'a G,
        x: &Xspace3D,
        potential: AP,           // Изменен тип параметра
        absorbing_potential: AB, // Изменен тип параметра
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
impl<'a, G, AP, AB> SSFM for SSFM3D<'a, G, AP, AB>
where
    AP: Fn([F; 3]) -> F + Send + Sync,
    AB: Fn([F; 3]) -> C + Send + Sync,
    G: GaugedEvolutionSSFM<3, AP, AB, WF = WaveFunction3D>,
{
    type WF = WaveFunction3D;

    fn time_step_evol(
        &mut self,
        wf: &mut WaveFunction3D,
        t: &mut Tspace,
        momentum_representation_callback: Option<&mut dyn FnMut(&WaveFunction3D, &Tspace)>,
    ) {
        self.fft_maker.modify_psi(wf);
        self.gauge.x_evol_half(
            self.particles,
            wf,
            t.current,
            t.dt,
            &self.potential,           // Добавлен &
            &self.absorbing_potential, // Добавлен &
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
                &self.potential,           // Добавлен &
                &self.absorbing_potential, // Добавлен &
            );
            t.current += t.dt;
        }

        self.fft_maker.do_fft(wf);
        self.gauge.p_evol(self.particles, wf, t.current, t.dt);

        // Вызов callback для волновой функцией в импульсном представлении
        if let Some(callback) = momentum_representation_callback {
            callback(wf, t);
        }

        self.fft_maker.do_ifft(wf);
        self.gauge.x_evol_half(
            self.particles,
            wf,
            t.current,
            t.dt,
            &self.potential,           // Добавлен &
            &self.absorbing_potential, // Добавлен &
        );
        self.fft_maker.demodify_psi(wf);
        t.current += t.dt;
    }
}
