// ================== ТЕСТ ==================
// fn main() {
//     let wf: WaveFunction2D = WaveFunction2D::init_from_hdf5("/home/denis/RustSSFM/RSSFM/src/out/br_linear_polarization_T2_2e1d/dim1/int_2e1d/out/time_evol/psi_x/psi_x_0.hdf5");
//     let calc = PurityCalculator::from_wavefunction(&wf);
//     let p = calc.purity_svd();
//     let p_direct = calc.purity_direct();
//     let (p_near, prob_near) = calc.purity_conditional_by_x2_cut(15.0); // пример x2_cut
//     println!("P={:?}", p);
//     println!("P_direct={:?}", p_direct);
//     println!("P_near={:?}", p_near);
//     println!("prob_near={:?}", prob_near);
//
//     // 1. Создаём калькулятор
//     let purity_calc = PurityCalculator::from_wavefunction(&wf);
//
//     // 2. Конфигурация сохранения орбиталей
//     let diag_cfg = OrbitalSaveConfig {
//         dir: PathBuf::from(format!("T2_diagnostics/t_{}", 5)),
//         n_orbitals: 6, // сохраняем первые 6 натуральных орбиталей
//     };
//
//     // 3. Запускаем диагностику
//     purity_calc
//         .write_diagnostic_with_orbitals_and_plots(
//             10,             // сколько n_k печатать в тексте
//             Some(diag_cfg), // сохраняем орбитали + графики
//         )
//         .expect("Purity diagnostic failed");
//
// }
use crate::config::{C, F}; // ваши алиасы из проекта
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

pub struct PurityCalculator<'a> {
    pub wf: &'a WaveFunction2D,
}

impl<'a> PurityCalculator<'a> {
    /// Создаёт калькулятор (ссылка на WaveFunction2D)
    pub fn from_wavefunction(wf: &'a WaveFunction2D) -> Self {
        Self { wf }
    }

    /// Нормировка волновой функции: проверяем и при необходимости нормируем
    /// Требование: sum_{i,j} |Psi_{ij}|^2 * dx1 * dx2 == 1
    fn ensure_normalized(&self, mut psi: Array2<C>) -> Array2<C> {
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        let mut norm_sq: F = 0.0;
        for val in psi.iter() {
            // val.norm_sqr() возвращает F (если C = Complex<F>)
            norm_sq += val.norm_sqr();
        }
        norm_sq *= dx1 * dx2;
        if norm_sq <= 0.0 {
            panic!("Wavefunction has zero norm");
        }
        if (norm_sq - 1.0).abs() > 1e-12 {
            let scale = 1.0 / norm_sq.sqrt();
            psi.mapv_inplace(|c| c * C::new(scale, 0.0));
        }
        psi
    }

    /// Построить матрицу A = sqrt(dx1) * Psi * sqrt(dx2)
    /// если `cols` указаны — берём только эти колонки (условная версия).
    fn build_A(&self, cols: Option<&[usize]>) -> Array2<C> {
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        let sdx1 = (dx1 as f64).sqrt() as F;
        let sdx2 = (dx2 as f64).sqrt() as F;
        let sdx1_c: C = C::new(sdx1, 0.0);
        let sdx2_c: C = C::new(sdx2, 0.0);

        let psi = self.wf.psi.clone();
        let psi = self.ensure_normalized(psi);

        match cols {
            Some(col_idxs) => {
                let n1 = psi.nrows();
                let ncols = col_idxs.len();
                let mut a = Array2::<C>::zeros((n1, ncols));
                for (out_col, &in_col) in col_idxs.iter().enumerate() {
                    let col = psi.column(in_col);
                    let mut targ = a.column_mut(out_col);
                    for (i, v) in col.indexed_iter() {
                        targ[i] = *v * sdx1_c * sdx2_c;
                    }
                }
                a
            }
            None => {
                // полный вариант: умножаем строки и колонки эффективно
                let mut a = psi;
                // умножаем все элементы на sqrt(dx1)*sqrt(dx2)
                let scale = sdx1_c * sdx2_c;
                a.mapv_inplace(|c| c * scale);
                a
            }
        }
    }

    /// purity через SVD: P = sum s_k^4
    pub fn purity_svd(&self) -> F {
        let a = self.build_A(None);

        // SVD: A = U * diag(s) * V^H
        let (_, s, _) = a.svd(false, false).expect("SVD failed");

        // P = sum_k s_k^4
        s.iter().map(|&si| si.powi(4)).sum()
    }

    /// Прямой способ: сначала rho = A * A^H, потом Tr(rho^2).
    /// Используйте лишь для проверки (медленнее/памятнозатратно).
    pub fn purity_direct(&self) -> F {
        let a = self.build_A(None);
        let a_conj = a.mapv(|c| c.conj());
        let rho = a.dot(&a_conj.t()); // rho = A * A^H
                                      // purity = Tr(rho * rho) = sum_{i,j} rho_{i,j} rho_{j,i}
        let mut purity: F = 0.0;
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let v = rho[(i, j)] * rho[(j, i)];
                purity += v.re; // чистота — вещественное число
            }
        }
        purity
    }

    /// Условная чистота: регион по второй координате задаётся порогом x2 such that |x2| < xcut
    /// Возвращает (purity_conditional, prob_region)
    pub fn purity_conditional_by_x2_cut(&self, x2_cut: F) -> (F, F) {
        let grid_x2 = &self.wf.x.grid[1];
        let mut cols: Vec<usize> = Vec::new();

        for (j, &x2) in grid_x2.iter().enumerate() {
            if x2.abs() <= x2_cut {
                cols.push(j);
            }
        }
        if cols.is_empty() {
            return (0.0, 0.0);
        }

        let a_r = self.build_A(Some(&cols));

        let (_, s, _) = a_r.svd(false, false).expect("SVD failed (conditional)");

        // prob = sum s^2
        let prob: F = s.iter().map(|&si| si.powi(2)).sum();
        if prob <= 0.0 {
            return (0.0, 0.0);
        }

        // P_cond = sum s^4 / (sum s^2)^2
        let sum_s4: F = s.iter().map(|&si| si.powi(4)).sum();
        let purity_cond = sum_s4 / (prob * prob);

        (purity_cond, prob)
    }

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
        let a = self.build_A(None);

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
}

impl<'a> PurityCalculator<'a> {
    /// Расширенная диагностика: текстовый отчёт + опциональная сохранение N орбиталей
    /// в формате WaveFunction1D + два графика:
    /// - orbitals.png: |phi_k(x)|^2 (несколько орбиталей, с легендой)
    /// - eigenvalues.png: log10(n_k) vs k
    pub fn write_diagnostic_with_orbitals_and_plots(
        &self,
        n_modes_txt: usize,
        save_cfg: Option<OrbitalSaveConfig>,
    ) -> IoResult<()> {
        // 1) Путь директории диагностики
        let dir_path: PathBuf = match &save_cfg {
            Some(cfg) => cfg.dir.clone(),
            None => PathBuf::from("diagnostics"),
        };
        create_dir_all(&dir_path).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create diagnostics dir {:?}: {:?}", dir_path, e),
            )
        })?;

        // 2) Открываем текстовый файл отчёта
        let txt_path = dir_path.join("diagnostic.txt");
        let mut txt_file = File::create(&txt_path)?;

        writeln!(txt_file, "Purity diagnostic report")?;
        writeln!(txt_file, "Generated at: {:?}\n", SystemTime::now())?;

        // 3) Базовая инфа по сетке
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        writeln!(txt_file, "Grid dx: dx1 = {:?}, dx2 = {:?}", dx1, dx2)?;
        writeln!(
            txt_file,
            "Grid sizes: n1 = {}, n2 = {}",
            self.wf.psi.nrows(),
            self.wf.psi.ncols()
        )?;
        writeln!(txt_file)?;

        // 4) Проверка нормировки
        let mut norm_sq: f64 = 0.0;
        for v in self.wf.psi.iter() {
            norm_sq += v.norm_sqr() as f64;
        }
        norm_sq *= (dx1 as f64) * (dx2 as f64);
        writeln!(
            txt_file,
            "Normalization check: sum |Psi|^2 dx1 dx2 = {:.12e}",
            norm_sq
        )?;
        writeln!(
            txt_file,
            "Note: code normalizes internally before SVD if needed.\n"
        )?;

        // 5) SVD: получаем U и s
        let a = self.build_A(None);
        let (u_opt, s, _vt_opt) = a.svd(true, false).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("SVD failed: {:?}", e))
        })?;

        writeln!(txt_file, "SVD done. number of modes = {}\n", s.len())?;

        // 6) Выписываем топ n_modes_txt
        let show = std::cmp::min(n_modes_txt, s.len());
        writeln!(txt_file, "Top {} modes:", show)?;
        writeln!(
            txt_file,
            "{:>4} {:>18} {:>18} {:>18}",
            "k", "s_k", "n_k = s_k^2", "n_k^2"
        )?;
        let mut sum_n: f64 = 0.0;
        let mut sum_n2: f64 = 0.0;
        for (k, &sk) in s.iter().enumerate() {
            let nk = (sk * sk) as f64;
            let nk2 = nk * nk;
            if k < show {
                writeln!(
                    txt_file,
                    "{:4} {:18.12e} {:18.12e} {:18.12e}",
                    k, sk, nk, nk2
                )?;
            }
            sum_n += nk;
            sum_n2 += nk2;
        }
        writeln!(txt_file)?;
        writeln!(txt_file, "Sum n_k = {:.12e}", sum_n)?;
        writeln!(txt_file, "Purity P = {:.12e}", sum_n2)?;
        writeln!(txt_file)?;

        writeln!(txt_file, "Interpretation notes:")?;
        writeln!(
            txt_file,
            "- n_k are eigenvalues of the reduced density matrix rho_1."
        )?;
        writeln!(txt_file, "- Purity P = sum n_k^2.")?;
        writeln!(txt_file)?;

        // 7) Если не требуется сохранять орбитали — завершаем
        if save_cfg.is_none() {
            writeln!(txt_file, "Natural orbitals: NOT saved (save_cfg = None).")?;
            txt_file.flush()?;
            return Ok(());
        }

        // 8) Сохраняем орбитали как WaveFunction1D-compatible .h5 + делаем графики
        let cfg = save_cfg.unwrap();
        let u = u_opt.expect("U matrix must be present to save natural orbitals");

        // создаём поддиректорию orbitals
        let orbitals_dir = cfg.dir.join("orbitals");
        create_dir_all(&orbitals_dir).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to create orbitals dir {:?}: {:?}", orbitals_dir, e),
            )
        })?;

        // подготовка для рисования
        let norb_avail = std::cmp::min(cfg.n_orbitals, u.ncols());
        let mut occs: Vec<f64> = Vec::with_capacity(norb_avail);
        let mut orbitals_complex: Vec<Array1<C>> = Vec::with_capacity(norb_avail);

        // x-сетка первой координаты
        let x_grid = self.wf.x.grid[0].clone();
        let n1 = x_grid.len();
        let x_min = x_grid[0];
        let x_max = x_grid[n1 - 1];

        // Сохраняем каждую орбиталь в отдельный .h5 файл (WaveFunction1D format)
        for k in 0..norb_avail {
            let col = u.column(k).to_owned(); // Array1<Complex<F>>
            let occ = (s[k] * s[k]) as f64;
            occs.push(occ);
            orbitals_complex.push(col.clone());

            // Создаём Xspace1D совместимо с WaveFunction1D
            let x0_val = x_grid[0];
            let dx1_val = self.wf.x.dx[0];
            let x1_space = Xspace1D {
                x0: [x0_val],
                dx: [dx1_val],
                n: [n1],
                grid: [x_grid.clone()],
            };

            // Создаём WaveFunction1D и сохраняем как .h5
            let wf1 = WaveFunction1D::new(col.clone(), x1_space);
            let fname = format!("orbital_{:03}.h5", k);
            let fpath = orbitals_dir.join(fname);
            wf1.save_as_hdf5(fpath.to_str().expect("invalid path")); // uses your method
        }

        writeln!(
            txt_file,
            "Saved {} orbitals into {:?}",
            norb_avail, orbitals_dir
        )?;
        writeln!(txt_file)?;

        // 9) Рисунок 1: |phi_k|^2 (линии)
        let orbitals_png = cfg.dir.join("orbitals.png");
        {
            let root = BitMapBackend::new(&orbitals_png, (1400, 900)).into_drawing_area();
            root.fill(&WHITE).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
            })?;

            // плотности
            let mut densities: Vec<Vec<f64>> = Vec::with_capacity(norb_avail);
            for orb in &orbitals_complex {
                densities.push(orb.iter().map(|z| z.norm_sqr() as f64).collect());
            }

            let mut y_max: f64 = densities
                .iter()
                .flat_map(|v| v.iter().cloned())
                .fold(0.0_f64, |a, b| a.max(b));
            if y_max <= 0.0 {
                y_max = 1.0;
            }

            let mut chart = ChartBuilder::on(&root)
                .caption(
                    "Natural orbitals |phi_k(x)|^2",
                    ("sans-serif", 24).into_font(),
                )
                .margin(10)
                .x_label_area_size(50)
                .y_label_area_size(80)
                .build_cartesian_2d(x_min..x_max, 0.0_f64..(y_max * 1.15))
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
                })?;

            chart
                .configure_mesh()
                .x_desc("x")
                .y_desc("|phi_k|^2")
                .draw()
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
                })?;

            // явная палитра
            let palette: Vec<RGBColor> = vec![
                RGBColor(228, 26, 28),
                RGBColor(55, 126, 184),
                RGBColor(77, 175, 74),
                RGBColor(152, 78, 163),
                RGBColor(255, 127, 0),
                RGBColor(255, 255, 51),
                RGBColor(166, 86, 40),
                RGBColor(247, 129, 191),
            ];

            for (idx, dens) in densities.iter().enumerate() {
                let color = palette[idx % palette.len()].clone();
                chart
                    .draw_series(LineSeries::new(
                        x_grid.iter().cloned().zip(dens.iter().cloned()),
                        &color,
                    ))
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Plot error: {:?}", e),
                        )
                    })?;

                // легенда — текст в координатах данных через chart.draw_series
                let label = format!("k={:} n={:.4e}", idx, occs[idx]);
                let legend_x = x_min + 0.02 * (x_max - x_min);
                let legend_y = y_max * (0.95 - 0.06 * idx as f64);
                chart
                    .draw_series(std::iter::once(Text::new(
                        label,
                        (legend_x, legend_y),
                        ("sans-serif", 15).into_font(),
                    )))
                    .map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Plot error: {:?}", e),
                        )
                    })?;
            }
        } // файл orbitals.png закрыт

        // 10) Рисунок 2: eigenvalues лог10(n_k) vs k
        let eigen_png = cfg.dir.join("eigenvalues.png");
        {
            let root = BitMapBackend::new(&eigen_png, (1000, 600)).into_drawing_area();
            root.fill(&WHITE).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
            })?;

            // построим логарифмированные значения (защищаем от нулей)
            let eps = 1e-16_f64;
            let logs: Vec<f64> = occs.iter().map(|&v| ((v.max(eps)).log10())).collect();

            // диапазон по x и y
            let n = logs.len();
            let x0 = 0i32;
            let x1 = (n as i32).saturating_sub(1);

            let y_min = logs.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut chart = ChartBuilder::on(&root)
                .caption("log10(occupation n_k) vs k", ("sans-serif", 24).into_font())
                .margin(10)
                .x_label_area_size(50)
                .y_label_area_size(80)
                .build_cartesian_2d(x0..(x1 + 1), y_min..y_max)
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
                })?;

            chart
                .configure_mesh()
                .x_desc("k")
                .y_desc("log10(n_k)")
                .draw()
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
                })?;

            chart
                .draw_series(LineSeries::new(
                    (0..logs.len()).map(|i| (i as i32, logs[i])),
                    &RGBColor(55, 126, 184),
                ))
                .map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, format!("Plot error: {:?}", e))
                })?;
        }

        writeln!(
            txt_file,
            "Orbitals and plots saved in directory: {:?}",
            cfg.dir
        )?;
        txt_file.flush()?;
        Ok(())
    }
}
