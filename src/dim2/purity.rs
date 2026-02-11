// ================== ПРИМЕР ==================
// use std::path::PathBuf;
//
// // предположим wf2e — ваша WaveFunction2D и it — текущий шаг
// let calc = PurityCalculator::new(&wf2e);
//
// сохранять 6 орбиталей в diagnostics/t_000150
// let diag_cfg = OrbitalSaveConfig {
//     dir: PathBuf::from(format!("diagnostics/t_{:06}", it)),
//     n_orbitals: 6,
// };
//
// // R — граница области, например R = 20.0
// let R: Option<F> = Some(20.0);
//
// // Запускаем диагностику: печать топ-12, сохранять орбитали, считать purity по R×R
// calc.write_diagnostic_with_orbitals_and_plots(12, Some(diag_cfg), R)
//     .expect("diagnostic failed");
// ================== ПРИМЕР ==================
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
    /// Создаём структуру на основе волновой функции
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

    /// Строим матрицу A = sqrt(dx1) * Psi * sqrt(dx2)
    /// если `cols` указаны — берём только эти колонки (то есть по заданной области).
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
    /// (медленнее/затратно).
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

// Вспомогательный перевод ошибок plotters/LAPACK -> std::io::Error
fn io_err<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))
}

impl<'a> PurityCalculator<'a> {
    /// Сохранить первые n_save натуральных орбиталей, получаемых из матрицы U и сингулярных чисел s.
    /// - `u` : Option<Array2<C>> от SVD для A (нужен U — иначе функция вернёт Err)
    /// - `s` : Array1<F> сингулярные числа
    /// - `dir` : директория, куда сохранять (создаётся)
    /// - `n_save` : сколько орбиталей сохранить
    /// Возвращает вектор occupation numbers (n_k = s_k^2) для сохранённых орбиталей.
    fn save_natural_orbitals_from_U(
        &self,
        u_opt: Option<Array2<C>>,
        s: &Array1<F>,
        dir: &PathBuf,
        n_save: usize,
        subset_row_indices: Option<&[usize]>, // если Some -> u относится к обрезанной сетке, используем соответствующую Xspace1D
    ) -> IoResult<Vec<F>> {
        // Проверка U
        let u = match u_opt {
            Some(m) => m,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "SVD did not return U matrix",
                ));
            }
        };

        create_dir_all(&dir).map_err(|e| io_err(e))?;

        // Собираем Xspace1D подходящий: если subset_row_indices задан — составляем обрезанную Xspace1D
        let (xspace, dx): (Xspace1D, F) = if let Some(rows) = subset_row_indices {
            // rows должны быть последовательными индексами, так как x>R на регулярной сетке
            let first = rows[0];
            let last = rows[rows.len() - 1];
            let grid_slice = self.wf.x.grid[0].slice(s![first..=last]).to_owned();
            let x0_val = grid_slice[0];
            let dx_val = self.wf.x.dx[0];
            let n_val = grid_slice.len();
            let xs = Xspace1D {
                x0: [x0_val],
                dx: [dx_val],
                n: [n_val],
                grid: [grid_slice],
            };
            (xs, dx_val)
        } else {
            // полная сетка
            let grid_full = self.wf.x.grid[0].clone();
            let x0_val = grid_full[0];
            let dx_val = self.wf.x.dx[0];
            let n_val = grid_full.len();
            let xs = Xspace1D {
                x0: [x0_val],
                dx: [dx_val],
                n: [n_val],
                grid: [grid_full],
            };
            (xs, dx_val)
        };

        // масштабирующий множитель для получения phi из U-столбца: phi = u_col / sqrt(dx)
        let dx_f64: f64 = dx as f64;
        let s_sqrt_dx: F = (dx_f64.sqrt()) as F;
        let scale = C::new(s_sqrt_dx, 0.0);

        let nsave = std::cmp::min(n_save, u.ncols());
        let mut occs: Vec<F> = Vec::with_capacity(nsave);

        for k in 0..nsave {
            // U column k
            let ucol = u.column(k).to_owned(); // length = number of rows (either full n1 or truncated)
                                               // Occupation number n_k = s_k^2
            let nk = (s[k] * s[k]) as F;
            occs.push(nk);

            // convert to physical orbital psi_i = ucol_i / sqrt(dx)
            let mut psi: Array1<C> = ucol.mapv(|c| c / scale);

            // create WaveFunction1D and normalize explicitly (safety)
            let mut wf1 = WaveFunction1D::new(psi, xspace.clone());
            wf1.normalization_by_1(); // ensure sum |psi|^2 dx = 1

            // save as hdf5: orbital file named orbital_###.h5
            let fname = format!("orbital_{:03}.h5", k);
            let fpath = dir.join(fname);
            wf1.save_as_hdf5(fpath.to_str().unwrap());

            // also save occupation number into same file under group "WaveFunction" as scalar `occupation`
            let occ_fname = format!("occupation_{:03}", k);
            write_scalar_to_hdf5(
                fpath.to_str().unwrap(),
                &occ_fname,
                Some("WaveFunction"),
                nk as f64,
            )
            .map_err(|e| io_err(e))?;
        }

        Ok(occs)
    }

    /// Переработанная функция purity_double_region:
    /// - если r_opt = None -> считает глобальную чистоту (через SVD на A) и возвращает (P, 1.0)
    /// - если r_opt = Some(R) -> считает условную чистоту для области {x1>R, x2>R} и возвращает (P_cond, p_R)
    /// (оставлена также для совместимости, но вы можете предпочесть использовать write_diagnostic... для сохранений)
    pub fn purity_double_region(&self, r_opt: Option<F>) -> (F, F) {
        // Построим A (весь)
        let a = self.build_A(None);
        // Если R не задан — считаем глобальную Purity через SVD на A
        if r_opt.is_none() {
            let (_, s, _) = a.svd(false, false).expect("SVD failed");
            let sum_s2: f64 = s.iter().map(|&si| (si * si) as f64).sum();
            let sum_s4: f64 = s.iter().map(|&si| (si * si * si * si) as f64).sum();
            // Для полной системы sum_s2 должен быть ~1
            let p = sum_s4 as F; // since sum_s2 == 1 -> P = sum_s4
            return (p, sum_s2 as F);
        }

        // Иначе: обрезанная область
        let r = r_opt.unwrap();
        // indices where x > R
        let x1 = &self.wf.x.grid[0];
        let x2 = &self.wf.x.grid[1];
        let rows: Vec<usize> = x1
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| {
                if (x as f64) > (r as f64) {
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
                if (x as f64) > (r as f64) {
                    Some(j)
                } else {
                    None
                }
            })
            .collect();

        if rows.is_empty() || cols.is_empty() {
            return (0.0 as F, 0.0 as F);
        }

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

        let (_u_r_opt, s_r, _v_r) = a_r.svd(false, false).expect("SVD failed on A_R");
        let sum_s2: f64 = s_r.iter().map(|&si| (si * si) as f64).sum();
        if sum_s2 <= 0.0 {
            return (0.0 as F, 0.0 as F);
        }
        let sum_s4: f64 = s_r.iter().map(|&si| (si * si * si * si) as f64).sum();
        let purity_cond = (sum_s4 / (sum_s2 * sum_s2)) as F;
        (purity_cond, sum_s2 as F)
    }

    /// Переписанная диагностика: сохраняет натуральные орбитали и occupation numbers
    /// как для всей области, так и (при указании R) для обрезанной области {x1>R,x2>R}.
    /// Параметры:
    ///  - n_save: сколько натуральных орбиталей сохранять (для каждого из full/region)
    ///  - cfg: Option<OrbitalSaveConfig> (директория и т.п.)
    ///  - r_opt: Option<F> (если Some(R) — вычисляем дополнительно для обрезанной области)
    pub fn write_diagnostic_with_orbitals_and_plots(
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
        let a_full = self.build_A(None);
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
}
