use crate::config::{C, F};
use crate::dim4::wave_function::{Representation, WaveFunction4D};
use crate::macros::check_path;
use crate::utils::hdf5_interface;
use crate::utils::logcolormap;
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

/// Двумерный срез четырехмерной волновой функции
pub struct WFSlice2D {
    pub psi_slice: Array2<C>,
    pub axes: [Array1<F>; 2],
    pub representation: Representation,
}

impl WFSlice2D {
    pub fn init_from_wf4d(
        wf: &WaveFunction4D,
        fixed_values: [Option<F>; 4], // None означает ось, по которой берется срез
    ) -> Self {
        // Определяем, импульсное представление или координатное
        let (grid, grid_first_elem, d_grid) = match wf.representation {
            Representation::Position => (&wf.x.grid, &wf.x.x0, &wf.x.dx),
            Representation::Momentum => (&wf.p.grid, &wf.p.p0, &wf.p.dp),
        };

        // Теперь используем переменные дальше в коде
        let mut fixed_indices: [Option<usize>; 4] = [None; 4];
        for i in 0..4 {
            if fixed_values[i].is_some() {
                let ind =
                    ((fixed_values[i].unwrap() - grid_first_elem[i]) / d_grid[i]).round() as usize;
                fixed_indices[i] = Some(ind);
            }
        }

        // Определяем, какие оси будут в срезе (те, для которых fixed_indices == None)
        let slice_axes: Vec<usize> = fixed_indices
            .iter()
            .enumerate()
            .filter(|(_, &idx)| idx.is_none())
            .map(|(i, _)| i)
            .collect();

        // Проверяем, что срез двумерный
        assert_eq!(slice_axes.len(), 2, "Slice must be 2D");

        // Создаем срез
        let slice = match fixed_indices {
            [None, None, Some(z), Some(w)] => s![.., .., z, w],
            [None, Some(y), None, Some(w)] => s![.., y, .., w],
            [None, Some(y), Some(z), None] => s![.., y, z, ..],
            [Some(x), None, None, Some(w)] => s![x, .., .., w],
            [Some(x), None, Some(z), None] => s![x, .., z, ..],
            [Some(x), Some(y), None, None] => s![x, y, .., ..],
            _ => panic!("Invalid slice configuration - exactly two axes must be None"),
        };

        // Применяем срез к пси-функции
        let psi_slice = wf.psi.slice(slice).to_owned();
        // создаем axes
        let axes_indices: Vec<usize> = (0..4).filter(|&i| fixed_values[i].is_none()).collect();
        let axes = [grid[axes_indices[0]].clone(), grid[axes_indices[1]].clone()];
        let representation = wf.representation;
        Self {
            psi_slice,
            axes,
            representation,
        }
    }

    pub fn plot_log(&self, path: &str, colorbar_limits: [F; 2]) {
        let [colorbar_min, colorbar_max] = colorbar_limits;
        let psi_slice_abs_sq = self.psi_slice.mapv(|elem| elem.norm_sqr());
        check_path!(path);
        logcolormap::plot_heatmap_logscale(
            &psi_slice_abs_sq,
            &self.axes[0],
            &self.axes[1],
            (colorbar_min, colorbar_max),
            path,
        )
        .unwrap();
    }

    pub fn save_as_hdf5(&self, path: &str) {
        check_path!(path);
        hdf5_interface::write_to_hdf5_complex(path, "psi_slice", None, &self.psi_slice).unwrap();
        hdf5_interface::create_str_data_attr(
            path,
            "psi_slice",
            None,
            "representation",
            self.representation.as_str(),
        )
        .unwrap();
        hdf5_interface::write_to_hdf5(path, "axis0", None, &self.axes[0]).unwrap();
        hdf5_interface::write_to_hdf5(path, "axis1", None, &self.axes[1]).unwrap();
    }
}
