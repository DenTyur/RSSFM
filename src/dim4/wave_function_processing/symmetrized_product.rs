use crate::config::{C, F};
use crate::dim2::wave_function::WaveFunction2D;
use crate::dim4::{
    space::{Pspace4D, Xspace4D},
    wave_function::WaveFunction4D,
};
use ndarray::{Array4, Zip};

/// Вычисление симметризованной двухэлектронной волновой функции без взаимодействия
/// из двух одноэлектронный волновых функций.
impl WaveFunction4D {
    pub fn new_symmetrized_product(wf1: &WaveFunction2D, wf2: &WaveFunction2D) -> Self {
        assert_eq!(
            wf1.representation, wf2.representation,
            "Одноэлектронные функции в разных представлениях"
        );
        let representation = wf1.representation;

        let n = wf1.psi.shape();
        let m = wf2.psi.shape();
        assert_eq!(n, m, "Волновые функции имеют разный размер");
        let mut wf2e: Array4<C> = Array4::zeros((n[0], n[1], m[0], m[1]));

        Zip::indexed(&mut wf2e).par_for_each(|(i0, i1, i2, i3), wf| {
            let term1 = wf1.psi[[i0, i1]] * wf2.psi[[i2, i3]];
            let term2 = wf2.psi[[i0, i1]] * wf1.psi[[i2, i3]];
            let normer: F = 2.0;
            *wf = (term1 + term2) / normer.sqrt();
        });

        let x = Xspace4D {
            x0: [wf1.x.x0[0], wf1.x.x0[1], wf2.x.x0[0], wf2.x.x0[1]],
            dx: [wf1.x.dx[0], wf1.x.dx[1], wf2.x.dx[0], wf2.x.dx[1]],
            n: [wf1.x.n[0], wf1.x.n[1], wf2.x.n[0], wf2.x.n[1]],
            grid: [
                wf1.x.grid[0].clone(),
                wf1.x.grid[1].clone(),
                wf2.x.grid[0].clone(),
                wf2.x.grid[1].clone(),
            ],
        };
        let p = Pspace4D::init(&x);
        Self {
            psi: wf2e,
            dpsi_d0: None,
            dpsi_d1: None,
            dpsi_d2: None,
            dpsi_d3: None,
            x,
            p,
            representation,
        }
    }
}
