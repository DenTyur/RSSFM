use crate::config::{C, F};
use crate::dim1::wave_function::WaveFunction1D;
use crate::dim2::{
    space::{Pspace2D, Xspace2D},
    wave_function::WaveFunction2D,
};
use ndarray::{Array2, Zip};

/// Вычисление симметризованной двухэлектронной волновой функции без взаимодействия
/// из двух одноэлектронный волновых функций.
impl WaveFunction2D {
    pub fn new_symmetrized_product(wf1: &WaveFunction1D, wf2: &WaveFunction1D) -> Self {
        assert_eq!(
            wf1.representation, wf2.representation,
            "Одноэлектронные функции в разных представлениях"
        );
        let representation = wf1.representation;

        let n = wf1.psi.len();
        let m = wf2.psi.len();
        assert_eq!(n, m, "Волновые функции имеют разный размер");
        let mut wf2e: Array2<C> = Array2::zeros((n, m));

        Zip::indexed(&mut wf2e).par_for_each(|(i, j), wf| {
            let term1 = wf1.psi[i] * wf2.psi[j];
            let term2 = wf2.psi[i] * wf1.psi[j];
            let normer: F = 2.0;
            *wf = (term1 + term2) / normer.sqrt();
        });

        let x = Xspace2D {
            x0: [wf1.x.x0[0], wf2.x.x0[0]],
            dx: [wf1.x.dx[0], wf2.x.dx[0]],
            n: [wf1.x.n[0], wf2.x.n[0]],
            grid: [wf1.x.grid[0].clone(), wf2.x.grid[0].clone()],
        };
        let p = Pspace2D::init(&x);
        Self {
            psi: wf2e,
            dpsi_dx: None,
            dpsi_dy: None,
            x,
            p,
            representation,
        }
    }
}
