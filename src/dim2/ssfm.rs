use super::fft_maker::FftMaker2D;
use super::space::Xspace2D;
use super::wave_function::WaveFunction2D;
use crate::common::particle::Particle;
use crate::common::tspace::Tspace;
use crate::config::{C, F};
use crate::traits::fft_maker::FftMaker;
use crate::traits::ssfm::{GaugedEvolutionSSFM, SSFM};
use crate::traits::wave_function::WaveFunction;

// pub struct SystemOfParticles2D {
//     pub number: usize,
//     pub masses: [F; 2],
//     pub charges: [F; 2],
// }
//
// impl SystemOfParticles2D {
//     const DIM: usize = 2;
//
//     pub fn new(particles: &[Particle]) -> Self {
//         // проверяем, влезают ли частицы в размерность
//         let mut total_dim: usize = 0;
//         for p in particles {
//             total_dim += p.dim;
//         }
//         assert!(
//             total_dim <= Self::DIM,
//             "Суммарная размерность частиц превышает 2"
//         );
//         // заполняем массы и заряды
//         let mut masses: [F; 2] = [0.0; 2];
//         let mut charges: [F; 2] = [0.0; 2];
//         match particles.len() {
//             0 => panic!("SystemOfParticles2D: Нет частиц!"),
//             // одна двумерная частица
//             1 => {
//                 masses = [particles[0].mass; 2];
//                 charges = [particles[0].charge; 2];
//             }
//             // две одномерные частицы
//             2 => {
//                 masses = [particles[0].mass, particles[1].mass];
//                 charges = [particles[0].charge, particles[1].charge];
//             }
//             _ => unreachable!("SystemOfParticles2D: перебор в размерности >2 !"),
//         }
//         Self { masses, charges }
//     }
// }

/// SSFM2D -- для одной двумерной частицы
pub struct SSFM2D<'a, G>
where
    G: GaugedEvolutionSSFM<2, WF = WaveFunction2D>,
{
    particles: &'a [Particle],
    potential: fn([F; 2]) -> F,
    absorbing_potential: fn([F; 2]) -> C,
    gauge: &'a G,
    fft_maker: FftMaker2D,
}

impl<'a, G> SSFM2D<'a, G>
where
    G: GaugedEvolutionSSFM<2, WF = WaveFunction2D>,
{
    pub fn new(
        particles: &'a [Particle],
        gauge: &'a G,
        x: &Xspace2D,
        potential: fn([F; 2]) -> F,
        absorbing_potential: fn([F; 2]) -> C,
    ) -> Self {
        let fft_maker = FftMaker2D::new(&x.n);
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
impl<'a, G> SSFM for SSFM2D<'a, G>
where
    G: GaugedEvolutionSSFM<2, WF = WaveFunction2D>,
{
    type WF = WaveFunction2D;

    fn time_step_evol(
        &mut self,
        wf: &mut WaveFunction2D,
        t: &mut Tspace,
        psi_p_save_path: Option<&str>,
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
            wf.save_as_npy(path).unwrap();
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
