//! # Вычисление чистоты (purity) одночастичной матрицы плотности
//!
//! Модуль предоставляет инструменты для анализа запутанности двухэлектронной волновой функции.
//! Основной вычисляемой величиной является **чистота** $\mathcal{P} = \operatorname{Tr}(\rho_1^2)$,
//! где $\rho_1$ – одночастичная матрица плотности, полученная интегрированием по координатам второго электрона.
//!
//! ## Теория
//!
//! Пусть задана двухэлектронная волновая функция $\Psi(x_1, x_2)$,
//! нормировка 
//! $$
//! \int |\Psi(x_1, x_2)|^2 dx_1 dx_2 = 1.
//! $$
//! Одночастичная матрица плотности определяется как
//! $$
//! \rho_1(x, x') = \int \Psi(x, x_2) \Psi^*(x', x_2) dx_2.
//! $$
//! Она является эрмитовой, положительно определённой и имеет единичный след
//! (поскольку $\int \rho_1(x, x) dx = 1$).
//!
//! Чистота (purity) вычисляется по формуле
//! $$
//! \mathcal{P} = \operatorname{Tr}(\rho_1^2) = \iint |\rho_1(x, x')|^2 dx dx' = \sum_k n_k^2,
//! $$
//! где $n_k$ – собственные числа $\rho_1$ (населенности орбиталей).
//! Для чистого состояния $\mathcal{P}=1$,
//! для максимально запутанного состояния в случай двух состояний $\mathcal{P}=0.5$.
//!
//! ### Связь с SVD волновой функции
//!
//! Рассмотрим оператор с ядром $A(x_1, x_2) = \sqrt{dx_1 dx_2}\, \Psi(x_1, x_2)$.
//! Сингулярное разложение (SVD) даёт
//! $$
//! A(x_1, x_2) = \sum_k s_k u_k(x_1) v_k^*(x_2),
//! $$
//! причём $\int |u_k(x)|^2 dx = \int |v_k(x)|^2 dx = 1$, $s_k \ge 0$.
//! Тогда одночастичная матрица плотности имеет собственные числа $n_k = s_k^2$,
//! и чистота равна $\mathcal{P} = \sum_k s_k^4$.
//!
//! ## Дискретизация на сетке
//!
//! Пусть волновая функция задана на равномерной сетке:
//! $\Psi_{ij} = \Psi(x_i, y_j)$ с шагами $\Delta x_1$, $\Delta x_2$.
//! Норма:
//! $$
//! \sum_{i,j} |\Psi_{ij}|^2 \Delta x_1 \Delta x_2 = 1.
//! $$
//! Определим матрицу $A$ размера $N_1 \times N_2$:
//! $$
//! A_{ij} = \sqrt{\Delta x_1 \Delta x_2}\; \Psi_{ij}.
//! $$
//! Сингулярные числа $s_k$ матрицы $A$ (полученные, например, с помощью `ndarray_linalg::svd`)
//! дают occupation numbers $n_k = s_k^2$, а чистота вычисляется как
//! $$
//! \mathcal{P} = \sum_{k=1}^{\min(N_1,N_2)} s_k^4.
//! $$
//! Этот подход не требует явного построения $\rho_1$.
//!
//! ## Чистота для области двойной ионизации
//!
//! Чтобы выделить вклад только той области, где оба электрона находятся далеко от ядра
//! ($|x_1| > R,\ |x_2| > R$), можно вычислить условную матрицу плотности,
//! ограничившись подмножеством сеточных узлов. Для этого:
//! * отбираются индексы $i$, для которых $x_i > R$, и $j$, для которых $y_j > R$;
//! * строится матрица $A_R$ размера $N_R^{(1)} \times N_R^{(2)}$ с элементами
//!   $(A_R)_{ij} = \sqrt{\Delta x_1 \Delta x_2}\; \Psi(x_i, y_j)$;
//! * выполняется SVD; находятся $s_k^{(R)}$, $n_k^{(R)} = (s_k^{(R)})^2$;
//! * вероятность найти оба электрона в области равна $p_R = \sum_k n_k^{(R)}$;
//! * условная чистота (purity внутри области) вычисляется как
//!   $$
//!   \mathcal{P}_{\text{cond}} = \frac{\sum_k (n_k^{(R)})^2}{p_R^2}.
//!   $$
//! Это даст степень запутанности именно в канале двойной ионизации.
//!
//! # Пример 1
//!
//! ```rust, no run 
//! use std::path::PathBuf;
//! 
//! // предположим wf2e — ваша WaveFunction2D и it — текущий шаг
//! let calc = PurityCalculator::new(&wf2e);
//! 
//! // сохранять 6 орбиталей в diagnostics/t_000150
//! let diag_cfg = OrbitalSaveConfig {
//!     dir: PathBuf::from(format!("diagnostics/t_{:06}", it)),
//!     n_orbitals: 6,
//! };
//! 
//! // R — граница области, например R = 20.0
//! let R: Option<F> = Some(20.0);
//! 
//! // Запускаем диагностику: печать топ-12, сохранять орбитали, считать purity по R×R
//! calc.write_diagnostic_with_orbitals_and_plots(12, Some(diag_cfg), R)
//!     .expect("diagnostic failed");
//! ```
//!
//! # Пример 2
//!
//! ```rust, no run 
//! fn main() {
//!     let wf: WaveFunction2D = WaveFunction2D::init_from_hdf5("/home/denis/RustSSFM/RSSFM/src/out/br_linear_polarization_T2_2e1d/dim1/int_2e1d/out/time_evol/psi_x/psi_x_0.hdf5");
//!     let calc = PurityCalculator::from_wavefunction(&wf);
//!     let p = calc.purity_svd();
//!     let p_direct = calc.purity_direct();
//!     let (p_near, prob_near) = calc.purity_conditional_by_x2_cut(15.0); // пример x2_cut
//!     println!("P={:?}", p);
//!     println!("P_direct={:?}", p_direct);
//!     println!("P_near={:?}", p_near);
//!     println!("prob_near={:?}", prob_near);
//! 
//!     // 1. Создаём калькулятор
//!     let purity_calc = PurityCalculator::from_wavefunction(&wf);
//! 
//!     // 2. Конфигурация сохранения орбиталей
//!     let diag_cfg = OrbitalSaveConfig {
//!         dir: PathBuf::from(format!("T2_diagnostics/t_{}", 5)),
//!         n_orbitals: 6, // сохраняем первые 6 натуральных орбиталей
//!     };
//! 
//!     // 3. Запускаем диагностику
//!     purity_calc
//!         .write_diagnostic_with_orbitals_and_plots(
//!             10,             // сколько n_k печатать в тексте
//!             Some(diag_cfg), // сохраняем орбитали + графики
//!         )
//!         .expect("Purity diagnostic failed");
//! }
//! ```

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

/// Калькулятор чистоты (purity) для двухэлектронной волновой функции.
///
/// Позволяет вычислить глобальную чистоту и условную чистоту в заданной области
/// (например, где оба электрона ионизованы). Также предоставляет диагностические функции,
/// сохраняющие натуральные орбитали и строящие графики.
pub struct PurityCalculator<'a> {
    /// Ссылка на двухэлектронную волновую функцию
    pub wf: &'a WaveFunction2D,
}

impl<'a> PurityCalculator<'a> {
    /// Создаёт новый экземпляр `PurityCalculator` на основе переданной волновой функции.
    pub fn from_wavefunction(wf: &'a WaveFunction2D) -> Self {
        Self { wf }
    }

    /// Нормирует волновую функцию (если необходимо) так, чтобы `sum_{i,j} |Psi_{ij}|^2 * dx1 * dx2 = 1`.
    ///
    /// Возвращает нормированную копию массива `psi`.
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

    /// Строит матрицу `A = sqrt(dx1) * Psi * sqrt(dx2)`
    ///
    /// Перед построением волновая функция нормируется.
    /// Возвращает массив `Array2<C>` формы `(n1, n2)`.
    pub fn build_full_A(&self) -> Array2<C> {
        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        let sdx1 = (dx1 as F).sqrt() as F;
        let sdx2 = (dx2 as F).sqrt() as F;
        let sdx1_c: C = C::new(sdx1, 0.0);
        let sdx2_c: C = C::new(sdx2, 0.0);

        let psi = self.wf.psi.clone();
        let psi = self.ensure_normalized(psi);

        let mut a = psi;
        let scale = sdx1_c * sdx2_c;
        a.mapv_inplace(|c| c * scale);
        a
    }

    /// Вычисляет purity глобально через SVD (наиболее стабильный способ).
    ///
    /// Возвращает `P = sum_k s_k^4`.
    pub fn purity_svd(&self) -> F {
        let a = self.build_full_A();
        let (_, s, _) = a.svd(false, false).expect("SVD failed");
        s.iter().map(|&si| si.powi(4)).sum()
    }

    /// Вычисляет purity прямым построением `rho = A * A^H` и взятием следа `Tr(rho^2)`.
    ///
    /// Более медленно и требует больше памяти, но полезно для проверки.
    pub fn purity_direct(&self) -> F {
        let a = self.build_full_A();
        let a_conj = a.mapv(|c| c.conj());
        let rho = a.dot(&a_conj.t()); // rho = A * A^H
        let mut purity: F = 0.0;
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let v = rho[(i, j)] * rho[(j, i)];
                purity += v.re;
            }
        }
        purity
    }
}

impl<'a> PurityCalculator<'a> {
    /// Вычисляет глобальную чистоту или условную чистоту в прямоугольной области `(x1 > R, x2 > R)`.
    ///
    /// * Если `r_opt = None`, возвращает `(P, 1.0)`, где `P` – глобальная чистота.
    /// * Если `r_opt = Some(R)`, возвращает `(P_cond, p_R)`, где `P_cond` – условная чистота,
    ///   а `p_R` – вероятность нахождения обоих электронов в области `x1 > R, x2 > R`.
    pub fn purity_double_region(&self, r_opt: Option<F>) -> (F, F) {
        let a = self.build_full_A();
        if r_opt.is_none() {
            let (_, s, _) = a.svd(false, false).expect("SVD failed");
            let sum_s2: F = s.iter().map(|&si| (si * si) as F).sum();
            let sum_s4: F = s.iter().map(|&si| (si * si * si * si) as F).sum();
            let p = sum_s4 as F;
            return (p, sum_s2 as F);
        }

        let r = r_opt.unwrap();
        let x1 = &self.wf.x.grid[0];
        let x2 = &self.wf.x.grid[1];
        let rows: Vec<usize> = x1
            .iter()
            .enumerate()
            .filter_map(|(i, &x)| if (x as F) > (r as F) { Some(i) } else { None })
            .collect();
        let cols: Vec<usize> = x2
            .iter()
            .enumerate()
            .filter_map(|(j, &x)| if (x as F) > (r as F) { Some(j) } else { None })
            .collect();

        if rows.is_empty() || cols.is_empty() {
            return (0.0 as F, 0.0 as F);
        }

        let dx1 = self.wf.x.dx[0];
        let dx2 = self.wf.x.dx[1];
        let scale = ((dx1 as F) * (dx2 as F)).sqrt() as F;
        let mut a_r = Array2::<C>::zeros((rows.len(), cols.len()));
        for (ii, &i) in rows.iter().enumerate() {
            for (jj, &j) in cols.iter().enumerate() {
                a_r[(ii, jj)] = self.wf.psi[(i, j)] * C::new(scale, 0.0);
            }
        }
        let (_u_r_opt, s_r, _v_r) = a_r.svd(false, false).expect("SVD failed on A_R");
        let sum_s2: F = s_r.iter().map(|&si| (si * si) as F).sum();
        if sum_s2 <= 0.0 {
            return (0.0 as F, 0.0 as F);
        }
        let sum_s4: F = s_r.iter().map(|&si| (si * si * si * si) as F).sum();
        let purity_cond = (sum_s4 / (sum_s2 * sum_s2)) as F;
        (purity_cond, sum_s2 as F)
    }
}
