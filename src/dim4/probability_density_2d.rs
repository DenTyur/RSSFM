use crate::config::F;
use crate::dim4::wave_function::{Representation, WaveFunction4D};
use crate::macros::check_path;
use crate::utils::hdf5_interface;
use crate::utils::logcolormap;
use ndarray::prelude::*;
use rayon::prelude::*;

use ndarray_npy::{ReadNpyExt, WriteNpyError, WriteNpyExt};
use num_complex::Complex;
use rayon::prelude::*;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::{BufRead, BufReader, Error, Write};

/// Плотность вероятности. Может работать как для импульсного, так и для координатного
/// представления
pub struct ProbabilityDensity2D {
    pub probability_density: Array2<F>,
    pub axes: [Array1<F>; 2],
    pub representation: Representation,
}

impl ProbabilityDensity2D {
    pub fn init_zeros(axes: [Array1<F>; 2], representation: Representation) -> Self {
        let n0 = axes[0].len();
        let n1 = axes[0].len();
        let probability_density: Array2<F> = Array2::zeros((n0, n1));
        Self {
            probability_density,
            axes,
            representation,
        }
    }

    pub fn compute_from_wf4d(
        wf: &WaveFunction4D,
        axes_inds: [usize; 2],
        integrate_axes_inds: [usize; 2],
        cut: Option<F>,
    ) -> Self {
        let axes: [Array1<F>; 2] = match wf.representation {
            Representation::Position => [
                wf.x.grid[axes_inds[0]].clone(),
                wf.x.grid[axes_inds[1]].clone(),
            ],
            Representation::Momentum => [
                wf.p.grid[axes_inds[0]].clone(),
                wf.p.grid[axes_inds[1]].clone(),
            ],
        };

        let (n0, n1) = (axes[0].len(), axes[1].len());
        let mut probability_density = Array2::zeros((n0, n1));

        let (d_axis0, d_axis1) = match wf.representation {
            Representation::Position => (
                wf.x.dx[integrate_axes_inds[0]],
                wf.x.dx[integrate_axes_inds[1]],
            ),
            Representation::Momentum => (
                wf.p.dp[integrate_axes_inds[0]],
                wf.p.dp[integrate_axes_inds[1]],
            ),
        };
        let volume_element = d_axis0 * d_axis1;

        let integrated_axes = match wf.representation {
            Representation::Position => [
                wf.x.grid[integrate_axes_inds[0]].clone(),
                wf.x.grid[integrate_axes_inds[1]].clone(),
            ],
            Representation::Momentum => [
                wf.p.grid[integrate_axes_inds[0]].clone(),
                wf.p.grid[integrate_axes_inds[1]].clone(),
            ],
        };
        let len_integrated_axes = [integrated_axes[0].len(), integrated_axes[1].len()];

        // Предварительно вычисляем маску для интегрирования
        let mask: Option<Array2<bool>> = cut.map(|cut_value| {
            Array2::from_shape_fn(
                (len_integrated_axes[0], len_integrated_axes[1]),
                |(k, l)| {
                    let coord1 = integrated_axes[0][k];
                    let coord2 = integrated_axes[1][l];
                    coord1.abs() > cut_value && coord2.abs() > cut_value
                },
            )
        });

        // Параллельное вычисление с итераторами
        probability_density
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..row.len() {
                    let slice = match (axes_inds, integrate_axes_inds) {
                        ([0, 1], [2, 3]) => wf.psi.slice(s![i, j, .., ..]),
                        ([0, 2], [1, 3]) => wf.psi.slice(s![i, .., j, ..]),
                        ([0, 3], [1, 2]) => wf.psi.slice(s![i, .., .., j]),
                        ([1, 2], [0, 3]) => wf.psi.slice(s![.., i, j, ..]),
                        ([1, 3], [0, 2]) => wf.psi.slice(s![.., i, .., j]),
                        ([2, 3], [0, 1]) => wf.psi.slice(s![.., .., i, j]),
                        _ => panic!("Невозможные оси!"),
                    };
                    let sum = if let Some(ref mask_arr) = mask {
                        slice
                            .iter()
                            .zip(mask_arr.iter())
                            .filter(|(_, &mask_val)| mask_val)
                            .map(|(psi_val, _)| psi_val.norm_sqr())
                            .sum::<F>()
                    } else {
                        slice.mapv(|c| c.norm_sqr()).sum()
                    };
                    row[j] = sum * volume_element;
                }
            });

        let representation = wf.representation;
        Self {
            probability_density,
            axes,
            representation,
        }
    }

    pub fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
        let [colorbar_min, colorbar_max] = colorbar_limits;
        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &self.probability_density,
            &self.axes[0],
            &self.axes[1],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }

    pub fn save_as_npy(&self, path: &str) -> Result<(), WriteNpyError> {
        check_path!(path);
        // Extract directory from path, default to current directory if none
        let dir_path = std::path::Path::new(path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::Path::new(".").to_path_buf());

        // Save sparsed wave function slice to NPY file
        let writer = BufWriter::new(File::create(path).map_err(|e| {
            WriteNpyError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create {}: {}", path, e),
            ))
        })?);

        self.probability_density.write_npy(writer)?;

        let ax0_path = dir_path.join("ax0.npy");
        let ax1_path = dir_path.join("ax1.npy");

        // Save sparsed wave function slice to NPY file
        let writer = BufWriter::new(File::create(ax0_path).map_err(|e| {
            WriteNpyError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create {}: {}", path, e),
            ))
        })?);

        self.axes[0].write_npy(writer)?;

        let writer = BufWriter::new(File::create(ax1_path).map_err(|e| {
            WriteNpyError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create {}: {}", path, e),
            ))
        })?);

        self.axes[1].write_npy(writer)?;

        Ok(())
    }

    pub fn save_as_hdf5(&self, path: &str) {
        check_path!(path);
        hdf5_interface::write_to_hdf5(path, "probability_density", None, &self.probability_density)
            .unwrap();
        hdf5_interface::create_str_data_attr(
            path,
            "probability_density",
            None,
            "representation",
            self.representation.as_str(),
        )
        .unwrap();
        hdf5_interface::write_to_hdf5(path, "axis0", None, &self.axes[0]).unwrap();
        hdf5_interface::write_to_hdf5(path, "axis1", None, &self.axes[1]).unwrap();
    }
}
