use super::logcolormap;
use crate::config::{C, F};
use crate::dim1::space::Xspace1D;
use crate::macros::check_path;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, Zip};
use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;

/// Структура для вычисления симметризованный двухэлектронной волновой функции без взаимодействия
/// из двух одноэлектронный волновых функций.
pub struct SymNonintWF2e1d {
    psi1: Array1<C>,
    psi2: Array1<C>,
    wf2e: Array2<C>,
    x: Xspace1D, // сетки psi1 и psi2 обязаны быть одинаковыми
}

impl SymNonintWF2e1d {
    pub fn new(psi1_path: &str, psi2_path: &str, x: Xspace1D) -> Self {
        let reader1 = File::open(psi1_path).unwrap();
        let psi1 = Array::<C, Ix1>::read_npy(reader1).unwrap();

        let reader2 = File::open(psi2_path).unwrap();
        let psi2 = Array::<C, Ix1>::read_npy(reader2).unwrap();

        let wf2e: Array2<C> = Array::zeros((psi1.len(), psi2.len()));

        Self {
            psi1,
            psi2,
            wf2e,
            x,
        }
    }

    pub fn compute_wf2e(&mut self) {
        let n = self.psi1.len();
        let m = self.psi2.len();
        assert_eq!(n, m, "Wave functions must have same length");

        Zip::indexed(&mut self.wf2e).par_for_each(|(i, j), wf| {
            let term1 = self.psi1[i] * self.psi2[j];
            let term2 = self.psi2[i] * self.psi1[j];
            let normer: F = 2.0;
            *wf = (term1 + term2) / normer.sqrt();
        });
    }

    pub fn save_as_npy(&self, path: &str) -> Result<(), WriteNpyError> {
        check_path!(path);
        let writer = BufWriter::new(File::create(path)?);
        self.wf2e.write_npy(writer)?;
        Ok(())
    }

    pub fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
        let mut a: Array2<F> = Array::zeros((self.x.n[0], self.x.n[0]));

        self.wf2e
            .axis_iter(Axis(0))
            .zip(a.axis_iter_mut(Axis(0)))
            .par_bridge()
            .for_each(|(psi_row, mut a_row)| {
                psi_row
                    .iter()
                    .zip(a_row.iter_mut())
                    .for_each(|(psi_elem, a_elem)| {
                        *a_elem = psi_elem.im.powi(2) + psi_elem.re.powi(2);
                    })
            });

        let [colorbar_min, colorbar_max] = colorbar_limits;

        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &a,
            &self.x.grid[0],
            &self.x.grid[0],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }
}
