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

// Вспомогательный перевод ошибок plotters/LAPACK -> std::io::Error
fn io_err<E: std::fmt::Debug>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e))
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
    pub fn ensure_normalized(&self, mut psi: Array2<C>) -> Array2<C> {
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
    pub fn build_full_A(&self) -> Array2<C> {
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        let sdx1 = (dx1 as f64).sqrt() as F;
        let sdx2 = (dx2 as f64).sqrt() as F;
        let sdx1_c: C = C::new(sdx1, 0.0);
        let sdx2_c: C = C::new(sdx2, 0.0);

        let psi = self.wf.psi.clone();
        let psi = self.ensure_normalized(psi);

        // умножаем строки и столбцы
        let mut a = psi;
        // умножаем все элементы на sqrt(dx1)*sqrt(dx2)
        let scale = sdx1_c * sdx2_c;
        a.mapv_inplace(|c| c * scale);
        a
    }

    /// purity через SVD: P = sum s_k^4
    pub fn purity_svd(&self) -> F {
        let a = self.build_full_A();

        // SVD: A = U * diag(s) * V^H
        let (_, s, _) = a.svd(false, false).expect("SVD failed");

        // P = sum_k s_k^4
        s.iter().map(|&si| si.powi(4)).sum()
    }

    /// Прямой способ: сначала rho = A * A^H, потом Tr(rho^2).
    /// (медленнее/затратно).
    pub fn purity_direct(&self) -> F {
        let a = self.build_full_A();
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
        let a = self.build_full_A();
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
}
