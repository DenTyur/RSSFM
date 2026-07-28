use crate::config::{C, F};
use crate::dim1::space::Xspace1D;
use crate::dim1::wave_function::WaveFunction1D;
use crate::dim2::wave_function::WaveFunction2D;
use crate::traits::wave_function::WaveFunction;
use crate::utils::hdf5_interface::*;
use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::svd::SVD;
use std::fs::File;
use std::io::{Result as IoResult, Write};
use std::time::SystemTime;

use crate::dim2::purity::PurityCalculator;
use crate::utils::hdf5_interface::{write_scalar_to_hdf5, write_to_hdf5_complex};
use ndarray::{Array1, Axis};
use plotters::prelude::*;
use std::fs::create_dir_all;
use std::path::Path;
use std::path::PathBuf;

pub struct OrbitalSaveConfig {
    pub dir: PathBuf, // директория, например PathBuf::from("diagnostics/timestep_150")
    pub n_orbitals: usize, // сколько орбиталей сохранять/рисовать
}

// Вспомогательный перевод ошибок plotters/LAPACK -> std::io::Error
fn io_err<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))
}

impl<'a> PurityCalculator<'a> {
    /// диагностика
    pub fn write_diagnostic(&self, path: &str, n_modes: usize) -> IoResult<()> {
        // Открываем файл для записи (перезаписываем если есть)
        let mut file = File::create(path)?;

        // Заголовок и время
        let now = SystemTime::now();
        writeln!(file, "Purity diagnostic report")?;
        writeln!(file, "Generated at: {:?}\n", now)?;

        // Базовая информация о сетке
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        writeln!(file, "Grid steps: dx1 = {:?}, dx2 = {:?}", dx1, dx2)?;
        writeln!(
            file,
            "Grid sizes: n1 = {}, n2 = {}",
            self.wf.psi.nrows(),
            self.wf.psi.ncols()
        )?;
        writeln!(file, "")?;

        // Проверка нормировки входной Psi (дискретная норма)
        let mut norm_sq: f64 = 0.0;
        for v in self.wf.psi.iter() {
            // v.norm_sqr() -> F (f32/f64), конвертируем в f64 для печати суммы
            norm_sq += v.norm_sqr() as f64;
        }
        norm_sq *= (dx1 as f64) * (dx2 as f64);
        writeln!(
            file,
            "Normalization check (sum |Psi|^2 * dx1 * dx2): {:.12e}",
            norm_sq
        )?;
        writeln!(
            file,
            "Note: code will normalize Psi internally before SVD if needed.\n"
        )?;

        // Построим A (метод build_A нормализует psi при необходимости)
        let a = self.build_full_A();

        // Выполним SVD: A = U * diag(s) * V^H
        let svd_res = a.svd(true, true);
        let (u_opt, s, vt_opt) = match svd_res {
            Ok(tuple) => tuple,
            Err(e) => {
                writeln!(file, "SVD failed: {:?}", e)?;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("SVD failed: {:?}", e),
                ));
            }
        };

        // Информация о возвращённых матрицах
        writeln!(
            file,
            "SVD returned: U present: {}, V^H present: {}\n",
            u_opt.is_some(),
            vt_opt.is_some()
        )?;

        // Количество найденных сингулярных чисел
        let total_modes = s.len();
        writeln!(file, "Total singular values (modes): {}\n", total_modes)?;

        // Выведем первые n_modes мод (или все, если меньше)
        let show = std::cmp::min(n_modes, total_modes);
        writeln!(
            file,
            "Top {} modes (index, s_k, n_k = s_k^2, contribution to purity = n_k^2):",
            show
        )?;
        writeln!(
            file,
            "{:>5} {:>18} {:>18} {:>18}",
            "k", "s_k", "n_k = s_k^2", "n_k^2"
        )?;

        let mut sum_n: f64 = 0.0;
        let mut sum_n2: f64 = 0.0;
        for (k, &sk) in s.iter().enumerate() {
            let nk = (sk * sk) as f64; // occupation number
            let nk2 = nk * nk; // contribution to purity
            if k < show {
                // Форматирование float: используем экспоненциальный формат для читаемости
                writeln!(file, "{:5} {:18.12e} {:18.12e} {:18.12e}", k, sk, nk, nk2)?;
            }
            sum_n += nk;
            sum_n2 += nk2;
        }

        writeln!(file, "\nSum occupations sum_k n_k  = {:.12e}", sum_n)?;
        writeln!(file, "Purity P = sum_k n_k^2         = {:.12e}", sum_n2)?;
        writeln!(file, "")?;

        // Небольшой аналитический блок-интерпретация
        writeln!(file, "Interpretation notes:")?;
        writeln!(
            file,
            "- s_k : сингулярные числа матрицы A = sqrt(dx1)*Psi*sqrt(dx2)."
        )?;
        writeln!(
            file,
            "- n_k = s_k^2 : собственные значения редуцированной плотности rho_1."
        )?;
        writeln!(file, "- Sum n_k (должна быть ~1) = {:.12e}", sum_n)?;
        writeln!(
            file,
            "- Purity P = sum n_k^2 : P=1 => одномодовая (почти чистая) одномерная плотность."
        );
        writeln!(file, "  Малые значения P (<<1) означают распределение заполняемости по многим модам (сильная смесь).")?;
        writeln!(file, "")?;

        writeln!(file, "Practical hints:")?;
        writeln!(
            file,
            "- Если sum n_k differs significantly from 1, проверьте нормировку Psi и factors dx."
        )?;
        writeln!(file, "- Типичный сценарий:")?;
        writeln!(
            file,
            "  * оба электрона в одной орбитали => n1 ~ 1 => P ~ 1"
        )?;
        writeln!(
            file,
            "  * Slater-детерминант двух ортонормированных орбиталей => n1 = n2 = 1/2 => P = 1/2"
        )?;
        writeln!(file, "")?;

        // Добавим небольшой блок с top-динамикой: если U есть — можем схематично записать нормы U-столбцов
        if let Some(u_mat) = u_opt {
            writeln!(
                file,
                "U matrix present: computing norms of first {} U-columns (||U_col||):",
                std::cmp::min(show, u_mat.ncols())
            )?;
            for col in 0..std::cmp::min(show, u_mat.ncols()) {
                let mut norm2: f64 = 0.0;
                for v in u_mat.column(col).iter() {
                    norm2 += v.norm_sqr() as f64;
                }
                writeln!(file, "U_col[{}] norm^2 = {:.12e}", col, norm2)?;
            }
            writeln!(file, "")?;
        } else {
            writeln!(
                file,
                "U matrix not returned by SVD (used faster SVD which didn't compute U)."
            )?;
        }

        // Завершающие замечания
        writeln!(file, "End of diagnostic.")?;
        writeln!(
            file,
            "You can compare `Purity = {:.12e}` with expected model values (1.0, 0.5, ...).",
            sum_n2
        )?;

        // flush и возврат Ok
        file.flush()?;
        Ok(())
    }

    /// Переписанная диагностика: сохраняет натуральные орбитали и occupation numbers
    /// как для всей области, так и (при указании R) для обрезанной области {x1>R,x2>R}.
    /// Параметры:
    ///  - n_save: сколько натуральных орбиталей сохранять (для каждого из full/region)
    ///  - cfg: Option<OrbitalSaveConfig> (директория и т.п.)
    ///  - r_opt: Option<F> (если Some(R) — вычисляем дополнительно для обрезанной области)
    pub fn last_diagnostic_with_orbitals_and_plots(
        &self,
        n_modes_txt: usize,
        save_cfg: Option<OrbitalSaveConfig>,
        r_opt: Option<F>,
    ) -> IoResult<()> {
        // Prepare directories
        let base_dir: PathBuf = match &save_cfg {
            Some(cfg) => cfg.dir.clone(),
            None => PathBuf::from("diagnostics"),
        };
        create_dir_all(&base_dir).map_err(|e| io_err(e))?;

        let txt_path = base_dir.join("diagnostic.txt");
        let mut txt = File::create(&txt_path)?;

        writeln!(txt, "Purity diagnostic report")?;
        writeln!(txt, "Generated at: {:?}\n", std::time::SystemTime::now())?;
        writeln!(
            txt,
            "Grid sizes: n1 = {}, n2 = {}",
            self.wf.psi.nrows(),
            self.wf.psi.ncols()
        )?;
        writeln!(txt)?;

        // -------- FULL: SVD on full A (with U) ----------
        let a_full = self.build_full_A();
        let (u_full_opt, s_full, _vt_full) = a_full.svd(true, false).map_err(|e| io_err(e))?;

        // occupation numbers n_k = s_k^2
        let n_full: Vec<f64> = s_full.iter().map(|&sk| (sk * sk) as f64).collect();
        let sum_n_full: f64 = n_full.iter().sum();
        let purity_full: f64 = n_full.iter().map(|&x| x * x).sum();

        writeln!(
            txt,
            "Full domain: sum n_k = {:.12e}, Purity P_full = {:.12e}",
            sum_n_full, purity_full
        )?;
        writeln!(txt, "First {} occupation numbers (full):", n_modes_txt)?;
        for (k, &nk) in n_full.iter().enumerate().take(n_modes_txt) {
            writeln!(txt, "  k={}  n_k = {:.12e}", k, nk)?;
        }
        writeln!(txt)?;

        // create orbitals directories
        let orbitals_dir = base_dir.join("orbitals");
        create_dir_all(&orbitals_dir).map_err(|e| io_err(e))?;
        let full_dir = orbitals_dir.join("full");
        create_dir_all(&full_dir).map_err(|e| io_err(e))?;

        // number to save
        let n_to_save = match &save_cfg {
            Some(c) => c.n_orbitals,
            None => 0,
        };

        // Save eigenvalues (full) as Array1<f64> in HDF5
        if n_full.len() > 0 {
            let arr_full = Array1::from_vec(n_full.clone());
            write_to_hdf5(
                &base_dir.join("eigenvalues_full.h5").to_string_lossy(),
                "eigenvalues_full",
                None,
                &arr_full,
            )
            .map_err(|e| io_err(e))?;
        }

        // Save first n_to_save orbitals (full) and also prepare densities for plotting
        let mut densities_full: Vec<Vec<f64>> = Vec::new();
        if n_to_save > 0 {
            // scale factor: phi = u_col / sqrt(dx)
            let dx = self.wf.x.dx[0];
            let scale = (dx.sqrt()) as F;
            let u_full = u_full_opt.clone().expect("U must be present for full SVD");
            let ncols = u_full.ncols();
            let save_count = std::cmp::min(n_to_save, ncols);
            for k in 0..save_count {
                // U column
                let ucol = u_full.column(k).to_owned(); // length = n1
                                                        // build phi = ucol / sqrt(dx)
                let phi: Array1<C> = ucol.mapv(|c| c / C::new(scale, 0.0));
                // build density for plotting
                let dens: Vec<f64> = phi.iter().map(|z| z.norm_sqr() as f64).collect();
                densities_full.push(dens);

                // Create WaveFunction1D and save
                let x_grid = self.wf.x.grid[0].clone();
                let x1_space = Xspace1D {
                    x0: [x_grid[0]],
                    dx: [dx],
                    n: [x_grid.len()],
                    grid: [x_grid.clone()],
                };
                let mut wf1 = WaveFunction1D::new(phi.clone(), x1_space.clone());
                // ensure normalized
                wf1.normalization_by_1();
                let fname = full_dir.join(format!("orbital_{:03}.h5", k));
                wf1.save_as_hdf5(fname.to_str().unwrap());
                // also write occupation scalar into same file
                write_scalar_to_hdf5(
                    fname.to_str().unwrap(),
                    &format!("occupation_{:03}", k),
                    Some("WaveFunction"),
                    (s_full[k] * s_full[k]) as f64,
                )
                .map_err(|e| io_err(e))?;
            }
        }

        // Plotting for FULL: orbitals_full.png and eigenvalues_full.png
        // Prepare x-grid for plotting
        let x_grid = self.wf.x.grid[0].clone();
        let x_min = x_grid[0];
        let x_max = x_grid[x_grid.len() - 1];

        // Colors/palette
        let palette: Vec<RGBColor> = vec![
            RGBColor(228, 26, 28),
            RGBColor(55, 126, 184),
            RGBColor(77, 175, 74),
            RGBColor(152, 78, 163),
            RGBColor(255, 127, 0),
            RGBColor(166, 86, 40),
            RGBColor(247, 129, 191),
            RGBColor(153, 153, 153),
        ];

        // orbitals plot (full)
        if !densities_full.is_empty() {
            let orbit_full_png = base_dir.join("orbitals_full.png");
            {
                let root = BitMapBackend::new(&orbit_full_png, (1200, 800)).into_drawing_area();
                root.fill(&WHITE).map_err(|e| io_err(e))?;

                let mut y_max = densities_full
                    .iter()
                    .flat_map(|v| v.iter())
                    .cloned()
                    .fold(0.0_f64, f64::max);
                if y_max <= 0.0 {
                    y_max = 1.0;
                }

                let mut chart = ChartBuilder::on(&root)
                    .caption("|phi_k(x)|^2 (full)", ("sans-serif", 20))
                    .margin(10)
                    .x_label_area_size(50)
                    .y_label_area_size(80)
                    .build_cartesian_2d(x_min..x_max, 0.0_f64..(y_max * 1.15))
                    .map_err(|e| io_err(e))?;

                chart
                    .configure_mesh()
                    .x_desc("x")
                    .y_desc("|phi_k|^2")
                    .draw()
                    .map_err(|e| io_err(e))?;

                let lw = 5u32; // thicker lines (2.5x)
                for (idx, dens) in densities_full.iter().enumerate() {
                    let col = palette[idx % palette.len()];
                    chart
                        .draw_series(LineSeries::new(
                            x_grid.iter().cloned().zip(dens.iter().cloned()),
                            col.stroke_width(lw),
                        ))
                        .map_err(|e| io_err(e))?;

                    // legend mark: short colored line and text
                    let y_legend = y_max * (0.95 - 0.06 * idx as f64);
                    let dx_leg = 0.03 * (x_max - x_min);
                    chart
                        .draw_series(LineSeries::new(
                            vec![
                                (x_min + 0.02 * (x_max - x_min), y_legend),
                                (x_min + 0.02 * (x_max - x_min) + dx_leg, y_legend),
                            ],
                            col.stroke_width(lw),
                        ))
                        .map_err(|e| io_err(e))?;
                    chart
                        .draw_series(std::iter::once(Text::new(
                            format!("k={} n={:.3e}", idx, (s_full[idx] * s_full[idx]) as f64),
                            (x_min + 0.02 * (x_max - x_min) + dx_leg * 1.3, y_legend),
                            ("sans-serif", 14).into_font(),
                        )))
                        .map_err(|e| io_err(e))?;
                }
            } // root dropped -> file closed
        }

        // eigenvalues plot (full) (log scale)
        {
            let eigen_full_png = base_dir.join("eigenvalues_full.png");
            let arr_nk: Vec<f64> = s_full.iter().map(|&sk| (sk * sk) as f64).collect();
            if !arr_nk.is_empty() {
                let logs: Vec<f64> = {
                    let eps = 1e-16;
                    arr_nk
                        .iter()
                        .map(|&v| (v.max(eps)).log10())
                        .collect::<Vec<_>>()
                };
                let ymin = logs.iter().cloned().fold(f64::INFINITY, f64::min);
                let ymax = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let root = BitMapBackend::new(&eigen_full_png, (1000, 600)).into_drawing_area();
                root.fill(&WHITE).map_err(|e| io_err(e))?;

                let mut chart = ChartBuilder::on(&root)
                    .caption("log10(n_k) (full)", ("sans-serif", 20))
                    .margin(10)
                    .x_label_area_size(50)
                    .y_label_area_size(80)
                    .build_cartesian_2d(0..(logs.len() as i32), ymin..ymax)
                    .map_err(|e| io_err(e))?;

                chart
                    .configure_mesh()
                    .x_desc("k")
                    .y_desc("log10(n_k)")
                    .draw()
                    .map_err(|e| io_err(e))?;

                chart
                    .draw_series(LineSeries::new(
                        logs.iter().enumerate().map(|(i, &v)| (i as i32, v)),
                        RGBColor(55, 126, 184).stroke_width(3),
                    ))
                    .map_err(|e| io_err(e))?;
            }
        }

        // ----------------- Now REGION diagnostics (if R given) --------------------
        if let Some(R) = r_opt {
            writeln!(txt, "Region diagnostics for R = {:.6}", R)?;
            // find indices
            let x1 = &self.wf.x.grid[0];
            let x2 = &self.wf.x.grid[1];
            let rows: Vec<usize> = x1
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if (x as f64) > (R as f64) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            let cols: Vec<usize> = x2
                .iter()
                .enumerate()
                .filter_map(|(j, &x)| {
                    if (x as f64) > (R as f64) {
                        Some(j)
                    } else {
                        None
                    }
                })
                .collect();

            if rows.is_empty() || cols.is_empty() {
                writeln!(txt, "  Region is empty (R too large). Skipping.")?;
            } else {
                // build A_R
                let dx1 = self.wf.x.dx[0];
                let dx2 = self.wf.x.dx[1];
                let scale = ((dx1 as f64) * (dx2 as f64)).sqrt() as F;
                let mut a_r = Array2::<C>::zeros((rows.len(), cols.len()));
                for (ii, &i) in rows.iter().enumerate() {
                    for (jj, &j) in cols.iter().enumerate() {
                        a_r[(ii, jj)] = self.wf.psi[(i, j)] * C::new(scale, 0.0);
                    }
                }

                let (u_r_opt, s_r, _vt_r) = a_r.svd(true, false).map_err(|e| io_err(e))?;
                let n_region: Vec<f64> = s_r.iter().map(|&sk| (sk * sk) as f64).collect();
                let sum_n_r: f64 = n_region.iter().sum();
                let sum_n2_r: f64 = n_region.iter().map(|&x| x * x).sum();
                // conditional purity:
                let p_r = sum_n_r;
                let p_cond = if p_r > 1e-14 {
                    sum_n2_r / (p_r * p_r)
                } else {
                    0.0
                };

                writeln!(txt, "  p_R = {:.12e}, P_cond = {:.12e}", p_r, p_cond)?;
                writeln!(txt, "  First {} occupation numbers (region):", n_modes_txt)?;
                for (k, &nk) in n_region.iter().enumerate().take(n_modes_txt) {
                    writeln!(txt, "    k={}  n_k = {:.12e}", k, nk)?;
                }

                // region dirs
                let region_dir = orbitals_dir.join(format!("region_R_{:.3}", R));
                create_dir_all(&region_dir).map_err(|e| io_err(e))?;
                // save eigenvalues (region)
                if !n_region.is_empty() {
                    let arr_region = Array1::from_vec(n_region.clone());
                    write_to_hdf5(
                        &region_dir.join("eigenvalues_region.h5").to_string_lossy(),
                        "eigenvalues_region",
                        None,
                        &arr_region,
                    )
                    .map_err(|e| io_err(e))?;
                }

                // save first n_to_save region orbitals and build densities
                let mut densities_region: Vec<Vec<f64>> = Vec::new();
                if n_to_save > 0 {
                    if let Some(u_r) = u_r_opt.clone() {
                        let dx = self.wf.x.dx[0];
                        let scale_phi = (dx.sqrt()) as F;
                        let save_count = std::cmp::min(n_to_save, u_r.ncols());
                        for k in 0..save_count {
                            let ucol = u_r.column(k).to_owned();
                            let phi = ucol.mapv(|c| c / C::new(scale_phi, 0.0));
                            let dens: Vec<f64> = phi.iter().map(|z| z.norm_sqr() as f64).collect();
                            densities_region.push(dens);

                            // make xspace corresponding to rows indices slice
                            let first = rows[0];
                            let last = rows[rows.len() - 1];
                            let grid_slice = self.wf.x.grid[0].slice(s![first..=last]).to_owned();
                            let x1_space = Xspace1D {
                                x0: [grid_slice[0]],
                                dx: [dx],
                                n: [grid_slice.len()],
                                grid: [grid_slice.clone()],
                            };
                            let mut wf1 = WaveFunction1D::new(phi.clone(), x1_space.clone());
                            wf1.normalization_by_1();
                            let fname = region_dir.join(format!("orbital_{:03}.h5", k));
                            wf1.save_as_hdf5(fname.to_str().unwrap());
                            write_scalar_to_hdf5(
                                fname.to_str().unwrap(),
                                &format!("occupation_{:03}", k),
                                Some("WaveFunction"),
                                (s_r[k] * s_r[k]) as f64,
                            )
                            .map_err(|e| io_err(e))?;
                        }
                    } else {
                        writeln!(txt, "  Warning: U not returned for region SVD; cannot save region orbitals.")?;
                    }
                }

                // plot region orbitals
                if !densities_region.is_empty() {
                    let orbit_region_png = base_dir.join("orbitals_region.png");
                    let root =
                        BitMapBackend::new(&orbit_region_png, (1200, 800)).into_drawing_area();
                    root.fill(&WHITE).map_err(|e| io_err(e))?;
                    let mut y_max = densities_region
                        .iter()
                        .flat_map(|v| v.iter())
                        .cloned()
                        .fold(0.0_f64, f64::max);
                    if y_max <= 0.0 {
                        y_max = 1.0;
                    }
                    let mut chart = ChartBuilder::on(&root)
                        .caption("|phi_k(x)|^2 (region)", ("sans-serif", 20))
                        .margin(10)
                        .x_label_area_size(50)
                        .y_label_area_size(80)
                        .build_cartesian_2d(x_min..x_max, 0.0_f64..(y_max * 1.15))
                        .map_err(|e| io_err(e))?;
                    chart
                        .configure_mesh()
                        .x_desc("x")
                        .y_desc("|phi_k|^2")
                        .draw()
                        .map_err(|e| io_err(e))?;
                    let lw = 5u32;
                    for (idx, dens) in densities_region.iter().enumerate() {
                        let col = palette[idx % palette.len()];
                        chart
                            .draw_series(LineSeries::new(
                                x_grid.iter().cloned().zip(dens.iter().cloned()),
                                col.stroke_width(lw),
                            ))
                            .map_err(|e| io_err(e))?;
                        let y_legend = y_max * (0.95 - 0.06 * idx as f64);
                        let dx_leg = 0.03 * (x_max - x_min);
                        chart
                            .draw_series(LineSeries::new(
                                vec![
                                    (x_min + 0.02 * (x_max - x_min), y_legend),
                                    (x_min + 0.02 * (x_max - x_min) + dx_leg, y_legend),
                                ],
                                col.stroke_width(lw),
                            ))
                            .map_err(|e| io_err(e))?;
                        chart
                            .draw_series(std::iter::once(Text::new(
                                format!("k={} n={:.3e}", idx, (s_r[idx] * s_r[idx]) as f64),
                                (x_min + 0.02 * (x_max - x_min) + dx_leg * 1.3, y_legend),
                                ("sans-serif", 14).into_font(),
                            )))
                            .map_err(|e| io_err(e))?;
                    }
                }

                // plot eigenvalues region (log)
                if !n_region.is_empty() {
                    let eigen_region_png = base_dir.join("eigenvalues_region.png");
                    let logs: Vec<f64> = {
                        let eps = 1e-16;
                        n_region
                            .iter()
                            .map(|&v| (v.max(eps)).log10())
                            .collect::<Vec<_>>()
                    };
                    let ymin = logs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let ymax = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let root =
                        BitMapBackend::new(&eigen_region_png, (1000, 600)).into_drawing_area();
                    root.fill(&WHITE).map_err(|e| io_err(e))?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("log10(n_k) (region)", ("sans-serif", 20))
                        .margin(10)
                        .x_label_area_size(50)
                        .y_label_area_size(80)
                        .build_cartesian_2d(0..(logs.len() as i32), ymin..ymax)
                        .map_err(|e| io_err(e))?;
                    chart
                        .configure_mesh()
                        .x_desc("k")
                        .y_desc("log10(n_k)")
                        .draw()
                        .map_err(|e| io_err(e))?;
                    chart
                        .draw_series(LineSeries::new(
                            logs.iter().enumerate().map(|(i, &v)| (i as i32, v)),
                            RGBColor(55, 126, 184).stroke_width(3),
                        ))
                        .map_err(|e| io_err(e))?;
                }
            } // end region not empty
        } // end if R

        writeln!(txt, "Diagnostics saved into {:?}", base_dir)?;
        txt.flush()?;
        Ok(())
    }

    /// Переписанная диагностика: сохраняет натуральные орбитали и occupation numbers
    /// как для всей области, так и (при указании R) для обрезанной области {x1>R,x2>R}.
    /// Дополнительно в diagnostic.txt записываются ненормированные n_k и нормированные lambda_k
    /// (lambda_k = n_k / sum_k n_k). Пояснения включены в файл.
    /// Параметры:
    ///  - n_modes_txt: сколько occupation numbers печатать в разделе summary
    ///  - save_cfg: Option<OrbitalSaveConfig> (директория и т.п.)
    ///  - r_opt: Option<F> (если Some(R) — вычисляем дополнительно для обрезанной области)
    pub fn diagnostic_with_orbitals_and_plots(
        &self,
        n_modes_txt: usize,
        save_cfg: Option<OrbitalSaveConfig>,
        r_opt: Option<F>,
    ) -> IoResult<()> {
        // Prepare directories
        let base_dir: PathBuf = match &save_cfg {
            Some(cfg) => cfg.dir.clone(),
            None => PathBuf::from("diagnostics"),
        };
        create_dir_all(&base_dir).map_err(|e| io_err(e))?;

        let txt_path = base_dir.join("diagnostic.txt");
        let mut txt = File::create(&txt_path)?;

        writeln!(txt, "Purity diagnostic report")?;
        writeln!(txt, "Generated at: {:?}\n", std::time::SystemTime::now())?;
        writeln!(
            txt,
            "Grid sizes: n1 = {}, n2 = {}",
            self.wf.psi.nrows(),
            self.wf.psi.ncols()
        )?;
        writeln!(txt)?;

        // -------- FULL: SVD on full A (with U) ----------
        let a_full = self.build_full_A();
        let (u_full_opt, s_full, _vt_full) = a_full.svd(true, false).map_err(|e| io_err(e))?;

        // occupation numbers n_k = s_k^2 (unnormalized; for full-domain sum ~ 1)
        let n_full: Vec<f64> = s_full.iter().map(|&sk| (sk * sk) as f64).collect();
        let sum_n_full: f64 = n_full.iter().sum();
        let purity_full: f64 = n_full.iter().map(|&x| x * x).sum();
        let p_raw_full: f64 = n_full.iter().map(|&x| x * x).sum(); // same as purity_full

        writeln!(
            txt,
            "Full domain: sum n_k = {:.12e}, Purity P_full = {:.12e}",
            sum_n_full, purity_full
        )?;
        // writeln!(txt, "First {} occupation numbers (full):", n_modes_txt)?;
        // for (k, &nk) in n_full.iter().enumerate().take(n_modes_txt) {
        //     writeln!(txt, "  k={}  n_k = {:.12e}", k, nk)?;
        // }

        writeln!(txt, "First {} singular values (full domain):", n_modes_txt)?;
        writeln!(txt, "  Columns: k | n_k (unnormalized) | n_k^(norm)")?;
        writeln!(txt, "  where:")?;
        writeln!(txt, "    n_k = s_k^2")?;
        writeln!(txt, "    sum_k n_k = {:.12e}", sum_n_full)?;
        writeln!(
            txt,
            "    Purity P = sum_k (n_k^(norm))^2 = {:.12e}",
            purity_full
        )?;
        writeln!(txt, "    n_k^(norm) = n_k / sum_k n_k")?;
        writeln!(txt)?;

        for (k, &nk) in n_full.iter().enumerate().take(n_modes_txt) {
            let nk_norm = if sum_n_full > 1e-14 {
                nk / sum_n_full
            } else {
                0.0
            };

            writeln!(txt, "  {:4}  {:18.10e}  {:18.10e}", k, nk, nk_norm)?;
        }
        writeln!(txt)?;
        //

        writeln!(txt)?;

        // create orbitals directories
        let orbitals_dir = base_dir.join("orbitals");
        create_dir_all(&orbitals_dir).map_err(|e| io_err(e))?;
        let full_dir = orbitals_dir.join("full");
        create_dir_all(&full_dir).map_err(|e| io_err(e))?;

        // number to save
        let n_to_save = match &save_cfg {
            Some(c) => c.n_orbitals,
            None => 0,
        };

        // Save eigenvalues (full) as Array1<f64> in HDF5
        if !n_full.is_empty() {
            let arr_full = Array1::from_vec(n_full.clone());
            write_to_hdf5(
                &base_dir.join("eigenvalues_full.h5").to_string_lossy(),
                "eigenvalues_full",
                None,
                &arr_full,
            )
            .map_err(|e| io_err(e))?;
        }

        // Save first n_to_save orbitals (full) and also prepare densities for plotting
        let mut densities_full: Vec<Vec<f64>> = Vec::new();
        if n_to_save > 0 {
            // scale factor: phi = u_col / sqrt(dx)
            let dx = self.wf.x.dx[0];
            let scale = (dx.sqrt()) as F;
            let u_full = u_full_opt.clone().expect("U must be present for full SVD");
            let ncols = u_full.ncols();
            let save_count = std::cmp::min(n_to_save, ncols);
            for k in 0..save_count {
                // U column
                let ucol = u_full.column(k).to_owned(); // length = n1
                                                        // build phi = ucol / sqrt(dx)
                let phi: Array1<C> = ucol.mapv(|c| c / C::new(scale, 0.0));
                // build density for plotting
                let dens: Vec<f64> = phi.iter().map(|z| z.norm_sqr() as f64).collect();
                densities_full.push(dens);

                // Create WaveFunction1D and save
                let x_grid = self.wf.x.grid[0].clone();
                let x1_space = Xspace1D {
                    x0: [x_grid[0]],
                    dx: [dx],
                    n: [x_grid.len()],
                    grid: [x_grid.clone()],
                };
                let mut wf1 = WaveFunction1D::new(phi.clone(), x1_space.clone());
                // ensure normalized
                wf1.normalization_by_1();
                let fname = full_dir.join(format!("orbital_{:03}.h5", k));
                wf1.save_as_hdf5(fname.to_str().unwrap());
                // also write occupation scalar into same file
                write_scalar_to_hdf5(
                    fname.to_str().unwrap(),
                    &format!("occupation_{:03}", k),
                    Some("WaveFunction"),
                    (s_full[k] * s_full[k]) as f64,
                )
                .map_err(|e| io_err(e))?;
            }
        }

        // Plotting for FULL: orbitals_full.png and eigenvalues_full.png
        // Prepare x-grid for plotting
        let x_grid = self.wf.x.grid[0].clone();
        let x_min = x_grid[0];
        let x_max = x_grid[x_grid.len() - 1];

        // Colors/palette
        let palette: Vec<RGBColor> = vec![
            RGBColor(228, 26, 28),
            RGBColor(55, 126, 184),
            RGBColor(77, 175, 74),
            RGBColor(152, 78, 163),
            RGBColor(255, 127, 0),
            RGBColor(166, 86, 40),
            RGBColor(247, 129, 191),
            RGBColor(153, 153, 153),
        ];

        // orbitals plot (full)
        if !densities_full.is_empty() {
            let orbit_full_png = base_dir.join("orbitals_full.png");
            {
                let root = BitMapBackend::new(&orbit_full_png, (1200, 800)).into_drawing_area();
                root.fill(&WHITE).map_err(|e| io_err(e))?;

                let mut y_max = densities_full
                    .iter()
                    .flat_map(|v| v.iter())
                    .cloned()
                    .fold(0.0_f64, f64::max);
                if y_max <= 0.0 {
                    y_max = 1.0;
                }

                let mut chart = ChartBuilder::on(&root)
                    .caption("|phi_k(x)|^2 (full)", ("sans-serif", 20))
                    .margin(10)
                    .x_label_area_size(50)
                    .y_label_area_size(80)
                    .build_cartesian_2d(x_min..x_max, 0.0_f64..(y_max * 1.15))
                    .map_err(|e| io_err(e))?;

                chart
                    .configure_mesh()
                    .x_desc("x")
                    .y_desc("|phi_k|^2")
                    .draw()
                    .map_err(|e| io_err(e))?;

                let lw = 5u32; // thicker lines (2.5x)
                for (idx, dens) in densities_full.iter().enumerate() {
                    let col = palette[idx % palette.len()];
                    chart
                        .draw_series(LineSeries::new(
                            x_grid.iter().cloned().zip(dens.iter().cloned()),
                            col.stroke_width(lw),
                        ))
                        .map_err(|e| io_err(e))?;

                    // legend mark: short colored line and text
                    let y_legend = y_max * (0.95 - 0.06 * idx as f64);
                    let dx_leg = 0.03 * (x_max - x_min);
                    chart
                        .draw_series(LineSeries::new(
                            vec![
                                (x_min + 0.02 * (x_max - x_min), y_legend),
                                (x_min + 0.02 * (x_max - x_min) + dx_leg, y_legend),
                            ],
                            col.stroke_width(lw),
                        ))
                        .map_err(|e| io_err(e))?;
                    chart
                        .draw_series(std::iter::once(Text::new(
                            format!("k={} n={:.3e}", idx, (s_full[idx] * s_full[idx]) as f64),
                            (x_min + 0.02 * (x_max - x_min) + dx_leg * 1.3, y_legend),
                            ("sans-serif", 14).into_font(),
                        )))
                        .map_err(|e| io_err(e))?;
                }
            } // root dropped -> file closed
        }

        // eigenvalues plot (full) (log scale)
        {
            let eigen_full_png = base_dir.join("eigenvalues_full.png");
            let arr_nk: Vec<f64> = s_full.iter().map(|&sk| (sk * sk) as f64).collect();
            if !arr_nk.is_empty() {
                let logs: Vec<f64> = {
                    let eps = 1e-16;
                    arr_nk
                        .iter()
                        .map(|&v| (v.max(eps)).log10())
                        .collect::<Vec<_>>()
                };
                let ymin = logs.iter().cloned().fold(f64::INFINITY, f64::min);
                let ymax = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let root = BitMapBackend::new(&eigen_full_png, (1000, 600)).into_drawing_area();
                root.fill(&WHITE).map_err(|e| io_err(e))?;

                let mut chart = ChartBuilder::on(&root)
                    .caption("log10(n_k) (full)", ("sans-serif", 20))
                    .margin(10)
                    .x_label_area_size(50)
                    .y_label_area_size(80)
                    .build_cartesian_2d(0..(logs.len() as i32), ymin..ymax)
                    .map_err(|e| io_err(e))?;

                chart
                    .configure_mesh()
                    .x_desc("k")
                    .y_desc("log10(n_k)")
                    .draw()
                    .map_err(|e| io_err(e))?;

                chart
                    .draw_series(LineSeries::new(
                        logs.iter().enumerate().map(|(i, &v)| (i as i32, v)),
                        RGBColor(55, 126, 184).stroke_width(3),
                    ))
                    .map_err(|e| io_err(e))?;
            }
        }

        // ----------------- Now REGION diagnostics (if R given) --------------------
        if let Some(R) = r_opt {
            writeln!(txt, "Region diagnostics for R = {:.6}", R)?;
            // find indices
            let x1 = &self.wf.x.grid[0];
            let x2 = &self.wf.x.grid[1];
            let rows: Vec<usize> = x1
                .iter()
                .enumerate()
                .filter_map(|(i, &x)| {
                    if (x as f64) > (R as f64) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            let cols: Vec<usize> = x2
                .iter()
                .enumerate()
                .filter_map(|(j, &x)| {
                    if (x as f64) > (R as f64) {
                        Some(j)
                    } else {
                        None
                    }
                })
                .collect();

            if rows.is_empty() || cols.is_empty() {
                writeln!(txt, "  Region is empty (R too large). Skipping.")?;
            } else {
                // build A_R
                let dx1 = self.wf.x.dx[0];
                let dx2 = self.wf.x.dx[1];
                let scale = ((dx1 as f64) * (dx2 as f64)).sqrt() as F;
                let mut a_r = Array2::<C>::zeros((rows.len(), cols.len()));
                for (ii, &i) in rows.iter().enumerate() {
                    for (jj, &j) in cols.iter().enumerate() {
                        a_r[(ii, jj)] = self.wf.psi[(i, j)] * C::new(scale, 0.0);
                    }
                }

                let (u_r_opt, s_r, _vt_r) = a_r.svd(true, false).map_err(|e| io_err(e))?;
                let n_region: Vec<f64> = s_r.iter().map(|&sk| (sk * sk) as f64).collect();
                let sum_n_r: f64 = n_region.iter().sum();
                let sum_n2_r: f64 = n_region.iter().map(|&x| x * x).sum();
                // conditional purity:
                let p_r = sum_n_r;
                let p_cond = if p_r > 1e-14 {
                    sum_n2_r / (p_r * p_r)
                } else {
                    0.0
                };

                writeln!(txt, "  p_R = {:.12e}, P_cond = {:.12e}", p_r, p_cond)?;
                // writeln!(txt, "  First {} occupation numbers (region):", n_modes_txt)?;
                // for (k, &nk) in n_region.iter().enumerate().take(n_modes_txt) {
                //     writeln!(txt, "    k={}  n_k = {:.12e}", k, nk)?;
                // }

                writeln!(
                    txt,
                    "  First {} singular values (double-ionization region):",
                    n_modes_txt
                )?;
                writeln!(txt, "  Columns: k | n_k (unnormalized) | n_k^(norm)")?;
                writeln!(txt, "  where:")?;
                writeln!(txt, "    n_k = s_k^2")?;
                writeln!(txt, "    sum_k n_k = p_R = {:.12e}", sum_n_r)?;
                writeln!(
                    txt,
                    "    Conditional purity P_cond = sum_k (n_k^(norm))^2 = {:.12e}",
                    p_cond
                )?;
                writeln!(txt, "    n_k^(norm) = n_k / p_R")?;
                writeln!(txt)?;

                for (k, &nk) in n_region.iter().enumerate().take(n_modes_txt) {
                    let nk_norm = if sum_n_r > 1e-14 { nk / sum_n_r } else { 0.0 };

                    writeln!(txt, "  {:4}  {:18.10e}  {:18.10e}", k, nk, nk_norm)?;
                }
                writeln!(txt)?;
                //

                // region dirs
                let region_dir = orbitals_dir.join(format!("region_R_{:.3}", R));
                create_dir_all(&region_dir).map_err(|e| io_err(e))?;
                // save eigenvalues (region)
                if !n_region.is_empty() {
                    let arr_region = Array1::from_vec(n_region.clone());
                    write_to_hdf5(
                        &region_dir.join("eigenvalues_region.h5").to_string_lossy(),
                        "eigenvalues_region",
                        None,
                        &arr_region,
                    )
                    .map_err(|e| io_err(e))?;
                }

                // save first n_to_save region orbitals and build densities
                let mut densities_region: Vec<Vec<f64>> = Vec::new();
                if n_to_save > 0 {
                    if let Some(u_r) = u_r_opt.clone() {
                        let dx = self.wf.x.dx[0];
                        let scale_phi = (dx.sqrt()) as F;
                        let save_count = std::cmp::min(n_to_save, u_r.ncols());
                        for k in 0..save_count {
                            let ucol = u_r.column(k).to_owned();
                            let phi = ucol.mapv(|c| c / C::new(scale_phi, 0.0));
                            let dens: Vec<f64> = phi.iter().map(|z| z.norm_sqr() as f64).collect();
                            densities_region.push(dens);

                            // make xspace corresponding to rows indices slice
                            let first = rows[0];
                            let last = rows[rows.len() - 1];
                            let grid_slice = self.wf.x.grid[0].slice(s![first..=last]).to_owned();
                            let x1_space = Xspace1D {
                                x0: [grid_slice[0]],
                                dx: [dx],
                                n: [grid_slice.len()],
                                grid: [grid_slice.clone()],
                            };
                            let mut wf1 = WaveFunction1D::new(phi.clone(), x1_space.clone());
                            wf1.normalization_by_1();
                            let fname = region_dir.join(format!("orbital_{:03}.h5", k));
                            wf1.save_as_hdf5(fname.to_str().unwrap());
                            write_scalar_to_hdf5(
                                fname.to_str().unwrap(),
                                &format!("occupation_{:03}", k),
                                Some("WaveFunction"),
                                (s_r[k] * s_r[k]) as f64,
                            )
                            .map_err(|e| io_err(e))?;
                        }
                    } else {
                        writeln!(txt, "  Warning: U not returned for region SVD; cannot save region orbitals.")?;
                    }
                }

                // plot region orbitals
                if !densities_region.is_empty() {
                    let orbit_region_png = base_dir.join("orbitals_region.png");
                    let root =
                        BitMapBackend::new(&orbit_region_png, (1200, 800)).into_drawing_area();
                    root.fill(&WHITE).map_err(|e| io_err(e))?;
                    let mut y_max = densities_region
                        .iter()
                        .flat_map(|v| v.iter())
                        .cloned()
                        .fold(0.0_f64, f64::max);
                    if y_max <= 0.0 {
                        y_max = 1.0;
                    }
                    let mut chart = ChartBuilder::on(&root)
                        .caption("|phi_k(x)|^2 (region)", ("sans-serif", 20))
                        .margin(10)
                        .x_label_area_size(50)
                        .y_label_area_size(80)
                        .build_cartesian_2d(x_min..x_max, 0.0_f64..(y_max * 1.15))
                        .map_err(|e| io_err(e))?;
                    chart
                        .configure_mesh()
                        .x_desc("x")
                        .y_desc("|phi_k|^2")
                        .draw()
                        .map_err(|e| io_err(e))?;
                    let lw = 5u32;
                    for (idx, dens) in densities_region.iter().enumerate() {
                        let col = palette[idx % palette.len()];
                        chart
                            .draw_series(LineSeries::new(
                                x_grid.iter().cloned().zip(dens.iter().cloned()),
                                col.stroke_width(lw),
                            ))
                            .map_err(|e| io_err(e))?;
                        let y_legend = y_max * (0.95 - 0.06 * idx as f64);
                        let dx_leg = 0.03 * (x_max - x_min);
                        chart
                            .draw_series(LineSeries::new(
                                vec![
                                    (x_min + 0.02 * (x_max - x_min), y_legend),
                                    (x_min + 0.02 * (x_max - x_min) + dx_leg, y_legend),
                                ],
                                col.stroke_width(lw),
                            ))
                            .map_err(|e| io_err(e))?;
                        chart
                            .draw_series(std::iter::once(Text::new(
                                format!("k={} n={:.3e}", idx, (s_r[idx] * s_r[idx]) as f64),
                                (x_min + 0.02 * (x_max - x_min) + dx_leg * 1.3, y_legend),
                                ("sans-serif", 14).into_font(),
                            )))
                            .map_err(|e| io_err(e))?;
                    }
                }

                // plot eigenvalues region (log)
                if !n_region.is_empty() {
                    let eigen_region_png = base_dir.join("eigenvalues_region.png");
                    let logs: Vec<f64> = {
                        let eps = 1e-16;
                        n_region
                            .iter()
                            .map(|&v| (v.max(eps)).log10())
                            .collect::<Vec<_>>()
                    };
                    let ymin = logs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let ymax = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let root =
                        BitMapBackend::new(&eigen_region_png, (1000, 600)).into_drawing_area();
                    root.fill(&WHITE).map_err(|e| io_err(e))?;
                    let mut chart = ChartBuilder::on(&root)
                        .caption("log10(n_k) (region)", ("sans-serif", 20))
                        .margin(10)
                        .x_label_area_size(50)
                        .y_label_area_size(80)
                        .build_cartesian_2d(0..(logs.len() as i32), ymin..ymax)
                        .map_err(|e| io_err(e))?;
                    chart
                        .configure_mesh()
                        .x_desc("k")
                        .y_desc("log10(n_k)")
                        .draw()
                        .map_err(|e| io_err(e))?;
                    chart
                        .draw_series(LineSeries::new(
                            logs.iter().enumerate().map(|(i, &v)| (i as i32, v)),
                            RGBColor(55, 126, 184).stroke_width(3),
                        ))
                        .map_err(|e| io_err(e))?;
                }
            } // end region not empty
        } // end if R

        writeln!(txt, "Diagnostics saved into {:?}", base_dir)?;
        txt.flush()?;
        Ok(())
    }
}
