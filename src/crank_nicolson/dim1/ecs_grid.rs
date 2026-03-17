//! Модуль, реализующий комплексную координатную сетку для метода внешнего комплексного масштабирования (ECS).
//!
//! # Физическая мотивация
//!
//! При численном решении нестационарного уравнения Шрёдингера на ограниченной области
//! необходимо поглощать волновую функцию на границах, чтобы избежать нефизичных отражений.
//! Метод внешнего комплексного масштабирования (Exterior Complex Scaling, ECS) достигает этого
//! путём аналитического продолжения координаты в комплексную плоскость вне некоторого радиуса $R_0$.
//!
//! Идея: в области $|x| > R_0$ координата $x$ заменяется на комплексную величину
//! $z = x + i f(x)$, где $f(x)$ монотонно растёт. Тогда уходящие волны $\sim e^{ipx}$
//! приобретают экспоненциально затухающий множитель $e^{-p\text{Im}z}$, а приходящие —
//️! экспоненциально растут, что позволяет отбросить их условием квадратичной интегрируемости.
//!
//! В данной реализации используется **линейное отображение** вне $R_0$:
//! $$
//! x(s) = \begin{cases}
//! s, & |s| \le R_0, \\
//! s + i\theta(|s|-R_0)\operatorname{sign}(s), & |s| > R_0,
//! \end{cases}
//! $$
//! где $s$ — действительная параметрическая координата, совпадающая с исходной равномерной сеткой,
//! а $\theta$ — угол масштабирования (параметр затухания). Такое отображение непрерывно,
//! но его первая производная терпит разрыв на границе. Однако для метода конечных разностей
//! это не является критичным, если правильно учесть метрику.
//!
//! Производные отображения:
//! - внутри $R_0$: $x' = 1$, $x'' = 0$;
//! - вне $R_0$: $x' = 1 + i\theta\operatorname{sign}(s)$, $x'' = 0$.
//!
//! # Структуры
//!
//! * [`ECSParams`] — хранит параметры масштабирования.
//! * [`ECSGrid1D`] — сетка, содержащая для каждого узла параметрическую координату $s$,
//!   комплексную физическую координату $x(s)$ и её производные $dx/ds$, $d^2x/ds^2$.

use crate::config::{C, F};
use crate::dim1::space::Xspace1D;
use ndarray::Array1;
use num_complex::Complex;
use std::f64::consts::PI;

/// Параметры внешнего комплексного масштабирования (ECS).
///
/// # Поля
/// * `r0` — радиус $R_0$, внутри которого масштабирование отсутствует ($|x| \le R_0$).
/// * `theta` — угол масштабирования $\theta$ (в радианах). Обычно выбирается в диапазоне 0.3–0.7.
/// * `width` — ширина внешней области по параметрической координате (используется для информации,
///   не влияет на расчёт сетки).
#[derive(Clone, Copy, Debug)]
pub struct ECSParams {
    pub r0: F,
    pub theta: F,
    pub width: F,
}

impl Default for ECSParams {
    fn default() -> Self {
        Self {
            r0: 40.0,
            theta: 0.5,
            width: 40.0,
        }
    }
}

/// Комплексная сетка для ECS, построенная на основе равномерной действительной сетки [`Xspace1D`].
///
/// Для каждого узла $s_j$ хранятся:
/// * `s[j]` — параметрическая координата $s_j$ (действительная);
/// * `x[j]` — физическая комплексная координата $x(s_j)$;
/// * `dxds[j]` — первая производная $\displaystyle \frac{dx}{ds}\big|_{s_j}$;
/// * `d2xds2[j]` — вторая производная $\displaystyle \frac{d^2x}{ds^2}\big|_{s_j}$;
/// * `params` — параметры ECS, использованные при построении.
///
/// Эти данные необходимы для вычисления дискретного оператора кинетической энергии
/// в координатах $x(s)$.
pub struct ECSGrid1D {
    pub s: Array1<F>,
    pub x: Array1<C>,
    pub dxds: Array1<C>,
    pub d2xds2: Array1<C>,
    pub params: ECSParams,
}

impl ECSGrid1D {
    /// Создаёт новую сетку на основе обычной действительной сетки `x_grid` и параметров ECS.
    ///
    /// # Аргументы
    /// * `x_grid` — исходная действительная сетка (её узлы становятся параметрическими $s$).
    /// * `params` — параметры ECS.
    ///
    /// # Возвращаемое значение
    /// Структура [`ECSGrid1D`] с заполненными полями `x`, `dxds`, `d2xds2`.
    ///
    /// # Математическая реализация
    ///
    /// Для каждого узла $s_j$:
    /// * Если $|s_j| \le R_0$:
    ///   $$ x = s_j,\quad \frac{dx}{ds}=1,\quad \frac{d^2x}{ds^2}=0. $$
    /// * Если $s_j > R_0$:
    ///   $$ x = s_j + i\theta (s_j-R_0), $$
    ///   $$ \frac{dx}{ds}=1 + i\theta,\quad \frac{d^2x}{ds^2}=0. $$
    /// * Если $s_j < -R_0$:
    ///   $$ x = s_j - i\theta (|s_j|-R_0) = s_j - i\theta (-s_j-R_0) = s_j + i\theta (s_j+R_0), $$
    ///   что с учётом знака можно записать единообразно:
    ///   $$ x = s_j + i\theta(|s_j|-R_0)\operatorname{sign}(s_j). $$
    ///   Производная: $\frac{dx}{ds}=1 + i\theta\operatorname{sign}(s_j)$.
    ///
    /// В коде для отрицательной стороны используется `sign = -1.0`, что даёт
    /// $x = s_j + i\cdot(-1)\cdot\theta\cdot d$ и производную $1 + i\cdot(-1)\cdot\theta$.
    ///
    /// # Пример
    /// ```
    /// use rssfm::dim1::space::Xspace1D;
    /// use rssfm::crank_nicolson::dim1::ecs_grid::{ECSGrid1D, ECSParams};
    ///
    /// let xspace = Xspace1D::new([-100.0], [0.1], [2001]);
    /// let params = ECSParams { r0: 40.0, theta: 0.5, width: 40.0 };
    /// let grid = ECSGrid1D::new(&xspace, params);
    /// println!("Размер сетки: {}", grid.n_nodes());
    /// println!("Первый узел: x = {:?}, dx/ds = {:?}", grid.x[0], grid.dxds[0]);
    /// ```
    pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
        let s = x_grid.grid[0].clone(); // берём действительные узлы как параметрическую координату
        let n = s.len();
        let mut x = Array1::<C>::zeros(n);
        let mut dxds = Array1::<C>::zeros(n);
        let mut d2xds2 = Array1::<C>::zeros(n);

        let tan_theta = params.theta.tan();
        let r0 = params.r0;
        let ds = x_grid.dx[0];

        for (i, &s_val) in s.iter().enumerate() {
            let abs_s = s_val.abs();
            let sign = if s_val >= 0.0 { 1.0 } else { -1.0 };

            if abs_s <= r0 {
                // Внутренняя область: x = s, производные = 1, 0
                x[i] = Complex::new(s_val, 0.0);
                dxds[i] = Complex::new(1.0, 0.0);
                d2xds2[i] = Complex::new(0.0, 0.0);
            } else {
                let d = abs_s - r0;
                // линейное: x = s + i * theta * d
                let f_val = d * params.theta; // tan(theta) можно заменить на theta, если theta мал
                x[i] = Complex::new(s_val, sign * f_val);

                let fp = params.theta; // производная d f/ds = theta
                let fpp = 0.0;

                dxds[i] = Complex::new(1.0, sign * fp);
                d2xds2[i] = Complex::new(0.0, 0.0);
            }
        }

        Self {
            s,
            x,
            dxds,
            d2xds2,
            params,
        }
    }

    /// Возвращает количество узлов сетки.
    ///
    /// # Пример
    /// ```
    /// # use rssfm::dim1::space::Xspace1D;
    /// # use rssfm::crank_nicolson::dim1::ecs_grid::{ECSGrid1D, ECSParams};
    /// # let xspace = Xspace1D::new([-100.0], [0.1], [2001]);
    /// # let params = ECSParams::default();
    /// # let grid = ECSGrid1D::new(&xspace, params);
    /// assert_eq!(grid.n_nodes(), 2001);
    /// ```
    pub fn n_nodes(&self) -> usize {
        self.s.len()
    }
}

// use crate::config::{C, F};
// use crate::dim1::space::Xspace1D;
// use ndarray::Array1;
// use num_complex::Complex;
// use std::f64::consts::PI;
//
// /// Параметры внешнего комплексного масштабирования (ECS)
// #[derive(Clone, Copy, Debug)]
// pub struct ECSParams {
//     /// радиус, внутри которого масштабирование отсутствует (|x| <= R0)
//     pub r0: F,
//     /// угол масштабирования в радианах (обычно 0.3–0.7)
//     pub theta: F,
//     /// ширина внешней области по параметрической координате s
//     /// (сколько добавить узлов за R0)
//     pub width: F,
// }
//
// impl Default for ECSParams {
//     fn default() -> Self {
//         Self {
//             r0: 40.0,
//             theta: 0.5,
//             width: 40.0,
//         }
//     }
// }
//
// /// Комплексная сетка для ECS.
// /// Параметрическая координата s (действительная) – это исходная сетка Xspace1D.
// /// Физическая координата x = s + i * f(s), где f(s) квадратична вне R0.
// pub struct ECSGrid1D {
//     /// параметрическая координата (действительная)
//     pub s: Array1<F>,
//     /// физическая комплексная координата
//     pub x: Array1<C>,
//     /// первая производная dx/ds в каждом узле
//     pub dxds: Array1<C>,
//     /// вторая производная d²x/ds²
//     pub d2xds2: Array1<C>,
//     /// параметры ECS (для информации)
//     pub params: ECSParams,
// }
//
// impl ECSGrid1D {
//     /// Создаёт новую сетку на основе обычной действительной сетки Xspace1D
//     /// и параметров ECS. Внутренняя область (|x| <= R0) остаётся неизменной,
//     /// внешняя область комплексифицируется.
//     pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
//         let s = x_grid.grid[0].clone(); // берём действительные узлы как параметрическую координату
//         let n = s.len();
//         let mut x = Array1::<C>::zeros(n);
//         let mut dxds = Array1::<C>::zeros(n);
//         let mut d2xds2 = Array1::<C>::zeros(n);
//
//         let tan_theta = params.theta.tan();
//         let r0 = params.r0;
//         let ds = x_grid.dx[0];
//
//         for (i, &s_val) in s.iter().enumerate() {
//             let abs_s = s_val.abs();
//             let sign = if s_val >= 0.0 { 1.0 } else { -1.0 };
//
//             if abs_s <= r0 {
//                 // Внутренняя область: x = s, производные = 1, 0
//                 x[i] = Complex::new(s_val, 0.0);
//                 dxds[i] = Complex::new(1.0, 0.0);
//                 d2xds2[i] = Complex::new(0.0, 0.0);
//             } else {
//                 let d = abs_s - r0;
//                 // линейное: x = s + i * theta * d
//                 let f_val = d * params.theta; // tan(theta) можно заменить на theta, если theta мал
//                 x[i] = Complex::new(s_val, sign * f_val);
//
//                 let fp = params.theta; // производная d f/ds = theta
//                 let fpp = 0.0;
//
//                 dxds[i] = Complex::new(1.0, sign * fp);
//                 d2xds2[i] = Complex::new(0.0, 0.0);
//             }
//         }
//
//         Self {
//             s,
//             x,
//             dxds,
//             d2xds2,
//             params,
//         }
//     }
//
//     /// Количество узлов сетки
//     pub fn n_nodes(&self) -> usize {
//         self.s.len()
//     }
// }
