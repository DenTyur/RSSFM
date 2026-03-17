//! Пропагатор Крэнка-Николсон для одномерного уравнения Шрёдингера с ECS.
//!
//! # Физическая постановка
//!
//! Рассматривается уравнение
//! $$ i\frac{\partial\psi}{\partial t} = \hat H(t)\psi,\quad
//!    \hat H(t) = -\frac12\frac{\partial^2}{\partial x^2} + V(x,t), $$
//! где $x$ — комплексная координата, определённая через ECS (см. модуль [`ecs_grid`]).
//! В методе Крэнка-Николсон временная эволюция аппроксимируется неявной схемой второго порядка:
//! $$ \frac{\psi^{n+1}-\psi^n}{\Delta t}
//!    = -i\hat H(t+\Delta t/2)\frac{\psi^{n+1}+\psi^n}{2}. $$
//! После преобразований получаем линейную систему
//! $$ \left(I + \frac{i\Delta t}{2}\hat H\right)\psi^{n+1}
//!    = \left(I - \frac{i\Delta t}{2}\hat H\right)\psi^n. $$
//!
//! Для мнимого времени ($\tau = it$) уравнение становится диффузионным:
//! $$ -\frac{\partial\psi}{\partial\tau} = \hat H\psi, $$
//! и схема принимает вид
//! $$ \left(I + \frac{\Delta\tau}{2}\hat H\right)\psi^{\tau+\Delta\tau}
//!    = \left(I - \frac{\Delta\tau}{2}\hat H\right)\psi^\tau. $$
//!
//! # Дискретизация по пространству
//!
//! На равномерной параметрической сетке $s_j = s_0 + j\Delta s$ (шаг $\Delta s$ совпадает с исходным
//! шагом по $x$) оператор кинетической энергии $\hat T = -\frac12\frac{d^2}{dx^2}$
//! аппроксимируется с использованием трёхточечного шаблона на неравномерной комплексной сетке $x(s)$.
//! Для внутреннего узла $j$ (индексы от 1 до $N-2$) используется формула:
//! $$
//! \frac{d^2\psi}{dx^2}\bigg|\_j \approx
//!    \frac{2}{x_{j+1}-x_{j-1}}\left(
//!        \frac{\psi_{j+1}-\psi_j}{x_{j+1}-x_j}
//!      - \frac{\psi_j-\psi_{j-1}}{x_j-x_{j-1}}
//!    \right).
//! $$
//! Тогда
//! $$
//! (\hat T\psi)\_j = -\frac12\frac{d^2\psi}{dx^2}\bigg|\_j
//!    = \alpha_j\psi_{j-1} + \beta_j\psi_j + \gamma_j\psi_{j+1},
//! $$
//! где коэффициенты $\alpha_j,\beta_j,\gamma_j$ вычисляются через комплексные расстояния
//! $dx_- = x_j-x_{j-1}$, $dx_+ = x_{j+1}-x_j$, $dx_{\text{avg}} = x_{j+1}-x_{j-1}$:
//! $$
//! \alpha_j = -\frac{1}{dx_{\text{avg}}dx_-},\qquad
//!    \beta_j  = \frac{1}{dx_{\text{avg}}}\left(\frac{1}{dx_+}+\frac{1}{dx_-}\right),\qquad
//!    \gamma_j = -\frac{1}{dx_{\text{avg}}dx_+}.
//! $$
//! (Множитель $-1/2$ уже включён в эти выражения после умножения на $-1/2$.)
//!
//! На границах ($j=0$ и $j=N-1$) используется условие Дирихле $\psi=0$, поэтому соответствующие
//! коэффициенты обнуляются (в системе это означает, что связь с отсутствующими соседями отсутствует).
//!
//! # Структуры
//!
//! * [`CrankNicolson1D`] — основной пропагатор, хранящий сетку, кинетические коэффициенты
//!   и рабочие массивы для метода прогонки.

use super::ecs_grid::{ECSGrid1D, ECSParams};
use crate::common::tspace::Tspace;
use crate::config::{C, F, I};
use crate::dim1::space::Xspace1D;
use crate::dim1::wave_function::WaveFunction1D;
use itertools::multizip;
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

/// Пропагатор Крэнка-Николсон для одномерной задачи с ECS.
///
/// # Поля
/// * `grid` — комплексная сетка [`ECSGrid1D`].
/// * `alpha, beta, gamma` — коэффициенты дискретного оператора кинетической энергии
///   (см. формулы выше).
/// * `c_prime, d_prime` — рабочие массивы для метода прогонки (размер равен числу узлов).
pub struct CrankNicolson1D {
    pub grid: ECSGrid1D,
    pub alpha: Array1<C>,
    pub beta: Array1<C>,
    pub gamma: Array1<C>,
    pub c_prime: Array1<C>,
    pub d_prime: Array1<C>,
}

impl CrankNicolson1D {
    // (Закомментированная предыдущая версия конструктора сохранена для истории)
    // pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
    //     ...
    // }

    /// Создаёт новый пропагатор, используя аппроксимацию второй производной на неравномерной сетке.
    ///
    /// # Аргументы
    /// * `x_grid` — исходная равномерная сетка (будет использована как параметрическая $s$).
    /// * `params` — параметры ECS.
    ///
    /// # Реализация
    ///
    /// 1. Строится комплексная сетка `grid` вызовом `ECSGrid1D::new`.
    /// 2. Для всех внутренних узлов ($1 \le j \le N-2$) вычисляются комплексные расстояния
    ///    `dx_minus`, `dx_plus`, `dx_avg`.
    /// 3. По формулам, приведённым выше, рассчитываются `alpha[j]`, `beta[j]`, `gamma[j]`.
    /// 4. Граничным узлам (`j=0` и `j=N-1`) присваиваются нулевые значения для связей с несуществующими
    ///    соседями (условие Дирихле).
    /// 5. Инициализируются рабочие массивы `c_prime` и `d_prime` нулями.
    ///
    /// # Пример
    /// ```
    /// use rssfm::dim1::space::Xspace1D;
    /// use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    /// use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    ///
    /// let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    /// let params = ECSParams { r0: 8.0, theta: 0.0, width: 30.0 };
    /// let propagator = CrankNicolson1D::new(&xspace, params);
    /// ```
    pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
        let grid = ECSGrid1D::new(x_grid, params);
        let n = grid.n_nodes();
        // Шаг по параметрической координате (постоянный)
        let ds = x_grid.dx[0];

        let mut alpha = Array1::<C>::zeros(n);
        let mut beta = Array1::<C>::zeros(n);
        let mut gamma = Array1::<C>::zeros(n);

        // Внутренние узлы (i = 1 .. n-2) используют центральную формулу на неравномерной сетке
        for i in 1..n - 1 {
            let x_prev = grid.x[i - 1];
            let x_cur = grid.x[i];
            let x_next = grid.x[i + 1];

            let dx_minus = x_cur - x_prev; // комплексное
            let dx_plus = x_next - x_cur; // комплексное
            let dx_avg = x_next - x_prev; // комплексное

            // Коэффициенты для второй производной d²/dx² (без множителя -1/2)
            let a = C::new(2.0, 0.0) / (dx_avg * dx_minus);
            let b = -C::new(2.0, 0.0) / dx_avg
                * (C::new(1.0, 0.0) / dx_plus + C::new(1.0, 0.0) / dx_minus);
            let c = C::new(2.0, 0.0) / (dx_avg * dx_plus);

            // Оператор кинетической энергии T = -1/2 d²/dx²
            alpha[i] = -C::new(0.5, 0.0) * a;
            beta[i] = -C::new(0.5, 0.0) * b;
            gamma[i] = -C::new(0.5, 0.0) * c;
        }

        // Граничные узлы: условие Дирихле (psi = 0 на границах)
        // Для i = 0 и i = n-1 обнуляем коэффициенты, чтобы в уравнении не было связи с внешними точками
        alpha[0] = C::new(0.0, 0.0);
        gamma[0] = C::new(0.0, 0.0); // для i=0 gamma не нужна, но оставим
        alpha[n - 1] = C::new(0.0, 0.0);
        gamma[n - 1] = C::new(0.0, 0.0);

        Self {
            grid,
            alpha,
            beta,
            gamma,
            c_prime: Array1::zeros(n),
            d_prime: Array1::zeros(n),
        }
    }

    /// Инициализирует волновую функцию значениями, заданными замыканием `init_fn`.
    ///
    /// Во внутренней области ($|s| \le R_0$) используется действительная координата `x_i.re`,
    /// и значение вычисляется как `init_fn(x_i.re)`. Во внешней области ($|s| > R_0$)
    /// волновая функция принудительно обнуляется. Это гарантирует, что начальное состояние
    /// локализовано внутри области, где координата действительна, что физически оправдано
    /// для связанных состояний.
    ///
    /// # Аргументы
    /// * `wf` — изменяемая волновая функция (её поле `psi` будет перезаписано).
    /// * `init_fn` — замыкание, принимающее действительную координату $x$ и возвращающее
    ///   комплексное значение $\psi(x)$.
    ///
    /// # Пример
    /// ```
    ///  use rssfm::dim1::wave_function::WaveFunction1D;
    ///  use rssfm::dim1::space::Xspace1D;
    ///  use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    ///  use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    ///  use num_complex::Complex;
    ///  let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    ///  let params = ECSParams::default();
    ///  let mut propagator = CrankNicolson1D::new(&xspace, params);
    ///  let mut wf = WaveFunction1D::init_gauss_1d(xspace.clone(), 0.0, 0.0, 1.0);
    /// propagator.init_wavefunction(&mut wf, |x| {
    ///     let normer = (2.0 * std::f64::consts::PI).powf(-0.25);
    ///     normer * (-x * x / 4.0).exp()
    /// });
    /// ```
    pub fn init_wavefunction(&self, wf: &mut WaveFunction1D, init_fn: impl Fn(F) -> C + Sync) {
        multizip((wf.psi.iter_mut(), self.grid.s.iter(), self.grid.x.iter()))
            .par_bridge()
            .for_each(|(psi_i, &s_i, &x_i)| {
                if s_i.abs() <= self.grid.params.r0 {
                    *psi_i = init_fn(x_i.re); // внутри R0 x действителен
                } else {
                    *psi_i = Complex::new(0.0, 0.0);
                }
            });
    }

    /// Вычисляет норму волновой функции с учётом якобиана преобразования $dx/ds$.
    ///
    /// В комплексных координатах интеграл нормы записывается как
    /// $$ \|\psi\|^2 = \int |\psi(s)|^2 \left|\frac{dx}{ds}\right| ds. $$
    /// Дискретный аналог:
    /// $$ \|\psi\|^2 \approx \sum_{j} |\psi_j|^2  \left|\frac{dx}{ds}\right|\_j  \Delta s, $$
    /// где $\Delta s$ — шаг по параметрической координате.
    ///
    /// # Аргументы
    /// * `wf` — волновая функция.
    ///
    /// # Возвращаемое значение
    /// Норма (положительное действительное число).
    ///
    /// # Пример
    /// ```
    ///  use rssfm::dim1::wave_function::WaveFunction1D;
    ///  use rssfm::dim1::space::Xspace1D;
    ///  use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    ///  use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    ///  let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    ///  let params = ECSParams::default();
    ///  let propagator = CrankNicolson1D::new(&xspace, params);
    ///  let mut wf = WaveFunction1D::init_gauss_1d(xspace, 0.0, 0.0, 1.0);
    ///  propagator.init_wavefunction(&mut wf, |x| (2.0*std::f64::consts::PI).powf(-0.25)*(-x*x/4.0).exp());
    ///  let norm = propagator.compute_norm(&wf);
    ///  println!("Норма с учётом якобиана: {}", norm);
    /// ```
    pub fn compute_norm(&self, wf: &WaveFunction1D) -> F {
        let ds = self.grid.s[1] - self.grid.s[0]; // шаг по параметрической координате
        self.grid
            .dxds
            .iter()
            .zip(wf.psi.iter())
            .map(|(&dxds_i, &psi_i)| psi_i.norm_sqr() * dxds_i.norm())
            .sum::<F>()
            * ds
    }

    /// Выполняет один шаг эволюции в реальном времени по схеме Крэнка-Николсон.
    ///
    /// # Аргументы
    /// * `wf` — изменяемая волновая функция (на входе $\psi^n$, на выходе $\psi^{n+1}$).
    /// * `t` — объект [`Tspace`], содержащий текущее время `t.current` и шаг `t.dt`.
    ///   После выполнения шага `t.current` увеличивается на `t.dt`.
    /// * `potential` — замыкание, возвращающее значение потенциала $V(x,t)$ в комплексной точке $x$
    ///   и в момент времени `t.current`.
    ///
    /// # Детали реализации
    ///
    /// 1. Из `t` извлекается шаг $\Delta t$.
    /// 2. Для каждого узла параллельно вычисляются:
    ///    - диагональные элементы матрицы $A = I + \frac{i\Delta t}{2}H$ и
    ///      вектора правой части $d = (I - \frac{i\Delta t}{2}H)\psi^n$.
    ///    - Используются предвычисленные коэффициенты `alpha`, `beta`, `gamma` и значение потенциала.
    /// 3. Полученная трёхдиагональная система $A\psi^{n+1}=d$ решается методом прогонки.
    /// 4. Решение копируется в `wf.psi`, время увеличивается.
    ///
    /// # Пример
    /// ```
    /// # use rssfm::dim1::space::Xspace1D;
    /// # use rssfm::dim1::wave_function::WaveFunction1D;
    /// # use rssfm::common::tspace::Tspace;
    /// # use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    /// # use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    /// # use num_complex::Complex;
    /// # let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    /// # let params = ECSParams { r0: 8.0, theta: 0.0, width: 30.0 };
    /// # let mut propagator = CrankNicolson1D::new(&xspace, params);
    /// # let mut wf = WaveFunction1D::init_gauss_1d(xspace, 0.0, 0.0, 1.0);
    /// # propagator.init_wavefunction(&mut wf, |x| (2.0*std::f64::consts::PI).powf(-0.25)*(-x*x/4.0).exp());
    /// let mut t = Tspace::new(0.0, 0.01, 1, 1000);
    /// let potential = |x: Complex<f64>, _t: f64| -> Complex<f64> {
    ///     -Complex::new(1.0, 0.0) / (Complex::new(2.0, 0.0) + x * x).sqrt()
    /// };
    /// propagator.propagate(&mut wf, &mut t, potential);
    /// ```
    pub fn propagate<F1>(&mut self, wf: &mut WaveFunction1D, t: &mut Tspace, potential: F1)
    where
        F1: Fn(C, F) -> C + Sync,
    {
        let n = self.grid.n_nodes();
        let psi_old = &wf.psi;
        let dt_c = Complex::new(t.dt, 0.0);
        let i_dt_half = I * dt_c * 0.5;

        let mut d = Array1::<C>::zeros(n);
        let mut a_diag = Array1::<C>::zeros(n);
        let mut a_off_low = Array1::<C>::zeros(n);
        let mut a_off_up = Array1::<C>::zeros(n);

        multizip((
            d.iter_mut(),
            a_diag.iter_mut(),
            a_off_low.iter_mut(),
            a_off_up.iter_mut(),
        ))
        .enumerate()
        .par_bridge()
        .for_each(|(i, (d_i, a_diag_i, a_low_i, a_up_i))| {
            let x_i = self.grid.x[i];
            let v = potential(x_i, t.current);

            let alpha_i = self.alpha[i];
            let beta_i = self.beta[i];
            let gamma_i = self.gamma[i];

            let h_diag = beta_i + v;
            let h_low = alpha_i;
            let h_up = gamma_i;

            *a_diag_i = Complex::new(1.0, 0.0) + i_dt_half * h_diag;
            *a_low_i = i_dt_half * h_low;
            *a_up_i = i_dt_half * h_up;

            // Проверка на потенциальные проблемы
            if a_diag_i.norm() < 1e-12 {
                println!("WARN: a_diag near zero at i={}, value={:?}", i, a_diag_i);
            }

            let psi_i = psi_old[i];
            let psi_l = if i > 0 {
                psi_old[i - 1]
            } else {
                Complex::new(0.0, 0.0)
            };
            let psi_r = if i < n - 1 {
                psi_old[i + 1]
            } else {
                Complex::new(0.0, 0.0)
            };

            let mut val = (Complex::new(1.0, 0.0) - i_dt_half * h_diag) * psi_i;
            if i > 0 {
                val -= i_dt_half * h_low * psi_l;
            }
            if i < n - 1 {
                val -= i_dt_half * h_up * psi_r;
            }
            *d_i = val;
        });

        let psi_new = self.solve_tridiagonal(&a_diag, &a_off_low, &a_off_up, &d);
        wf.psi.assign(&psi_new);
        t.current += t.dt
    }
    /// Выполняет один шаг эволюции в реальном времени в velocity gauge.
    ///
    /// # Аргументы
    /// * `wf` — изменяемая волновая функция.
    /// * `t` — объект времени, содержащий текущее время и шаг `dt`.
    /// * `atomic_potential` — замыкание, возвращающее атомный потенциал \(V_{\text{atom}}(x)\) в комплексной точке \(x\).
    /// * `vector_potential` — замыкание, возвращающее векторный потенциал \(A(t)\) в момент времени \(t\).
    ///
    /// # Математическая основа
    /// Гамильтониан в velocity gauge (после удаления \(A^2/2\) унитарным преобразованием):
    /// $$ \hat H = -\frac12 \frac{d^2}{dx^2} + iA(t)\frac{d}{dx} + V_{\text{atom}}(x). $$
    /// Дискретизация на неравномерной комплексной сетке \(x_j\):
    /// - Оператор второй производной даёт трёхточечные коэффициенты \(\alpha_j,\beta_j,\gamma_j\) (из конструктора).
    /// - Оператор первой производной аппроксимируется центральной разностью:
    ///   $$ \left.\frac{d\psi}{dx}\right|_j \approx \frac{\psi_{j+1} - \psi_{j-1}}{2\Delta x_j}, $$
    ///   где \(\Delta x_j = (x_{j+1} - x_{j-1})/2\) (комплексное).
    ///   Его вклад в гамильтониан: \(iA(t)\frac{d\psi}{dx}\) даёт добавки к недиагональным элементам:
    ///   $$ p_j^- = -\frac{iA}{2\Delta x_j} \quad\text{(коэффициент при }\psi_{j-1}\text{)}, $$
    ///   $$ p_j^+ = +\frac{iA}{2\Delta x_j} \quad\text{(коэффициент при }\psi_{j+1}\text{)}. $$
    ///
    /// Полные коэффициенты для узла \(j\):
    /// $$ \alpha_j^{\text{tot}} = \alpha_j + p_j^-,\quad \beta_j^{\text{tot}} = \beta_j,\quad \gamma_j^{\text{tot}} = \gamma_j + p_j^+. $$
    ///
    /// Схема Крэнка-Николсон:
    /// $$ \left(I + \frac{i\Delta t}{2}\hat H\right)\psi^{n+1} = \left(I - \frac{i\Delta t}{2}\hat H\right)\psi^n. $$
    /// Приводит к трёхдиагональной системе \(A\psi^{n+1} = d\), где
    /// $$ A_{j,j-1} = \frac{i\Delta t}{2}\alpha_j^{\text{tot}},\quad
    ///    A_{j,j}   = 1 + \frac{i\Delta t}{2}(\beta_j^{\text{tot}} + V_{\text{atom}}(x_j)),\quad
    ///    A_{j,j+1} = \frac{i\Delta t}{2}\gamma_j^{\text{tot}}, $$
    /// $$ d_j = \psi_j^n - \frac{i\Delta t}{2}\bigl[\alpha_j^{\text{tot}}\psi_{j-1}^n + (\beta_j^{\text{tot}}+V_{\text{atom}}(x_j))\psi_j^n + \gamma_j^{\text{tot}}\psi_{j+1}^n\bigr]. $$
    ///
    /// # Примечания
    /// - На границах (\(j=0\) и \(j=N-1\)) вклад от первой производной обнуляется, так как там \(\psi=0\).
    /// - Векторный потенциал \(A(t)\) может быть комплексным, но обычно действителен.
    /// - Метод совместим с ECS: все величины, включая \(\Delta x_j\), комплексны, что обеспечивает правильное затухание уходящих волн.
    ///
    /// # Пример
    /// ```
    /// # use rssfm::dim1::space::Xspace1D;
    /// # use rssfm::dim1::wave_function::WaveFunction1D;
    /// # use rssfm::common::tspace::Tspace;
    /// # use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    /// # use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    /// # let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    /// # let params = ECSParams { r0: 8.0, theta: 0.0, width: 30.0 };
    /// # let mut propagator = CrankNicolson1D::new(&xspace, params);
    /// # let mut wf = WaveFunction1D::init_gauss_1d(xspace, 0.0, 0.0, 1.0);
    /// # propagator.init_wavefunction(&mut wf, |x| (2.0*std::f64::consts::PI).powf(-0.25)*(-x*x/4.0).exp());
    /// let mut t = Tspace::new(0.0, 0.01, 1, 1000);
    /// let atomic = |x: Complex<f64>| -> Complex<f64> {
    ///     -Complex::new(1.0, 0.0) / (Complex::new(2.0, 0.0) + x * x).sqrt()
    /// };
    /// let a_field = |tau: f64| -> Complex<f64> {
    ///     // A(t) = -E0/ω * sin(ωt) для поля E(t)=E0 cos(ωt)
    ///     let e0 = 0.05;
    ///     let omega = 0.057;
    ///     -Complex::new(e0/omega * (omega * tau).sin(), 0.0)
    /// };
    /// propagator.propagate_velocity(&mut wf, &mut t, atomic, a_field);
    /// ```
    pub fn propagate_velocity<F1, F2>(
        &mut self,
        wf: &mut WaveFunction1D,
        t: &mut Tspace,
        total_potential: F1,
        vector_potential: F2,
    ) where
        F1: Fn(C, F) -> C + Sync,
        F2: Fn(F) -> C + Sync,
    {
        let n = self.grid.n_nodes();
        let psi_old = &wf.psi;
        let dt_c = Complex::new(t.dt, 0.0);
        let i_dt_half = I * dt_c * 0.5;
        let A_t = vector_potential(t.current); // комплексное значение векторного потенциала

        let mut d = Array1::<C>::zeros(n);
        let mut a_diag = Array1::<C>::zeros(n);
        let mut a_off_low = Array1::<C>::zeros(n);
        let mut a_off_up = Array1::<C>::zeros(n);

        multizip((
            d.iter_mut(),
            a_diag.iter_mut(),
            a_off_low.iter_mut(),
            a_off_up.iter_mut(),
        ))
        .enumerate()
        .par_bridge()
        .for_each(|(i, (d_i, a_diag_i, a_low_i, a_up_i))| {
            let x_i = self.grid.x[i];
            let v_atomic = total_potential(x_i, t.current);

            // Вычисляем вклад от iA d/dx
            let (p_low, p_up) = if i > 0 && i < n - 1 {
                // Комплексный шаг для производной: (x_{i+1} - x_{i-1})/2
                let dx_center = (self.grid.x[i + 1] - self.grid.x[i - 1]) * 0.5;
                let p_low = -I * A_t / (2.0 * dx_center);
                let p_up = I * A_t / (2.0 * dx_center);
                (p_low, p_up)
            } else {
                (C::new(0.0, 0.0), C::new(0.0, 0.0))
            };

            let alpha_i = self.alpha[i] + p_low; // обратите внимание: p_low уже с правильным знаком для H
            let gamma_i = self.gamma[i] + p_up;
            let beta_i = self.beta[i];

            let h_diag = beta_i + v_atomic;
            let h_low = alpha_i;
            let h_up = gamma_i;

            *a_diag_i = C::new(1.0, 0.0) + i_dt_half * h_diag;
            *a_low_i = i_dt_half * h_low;
            *a_up_i = i_dt_half * h_up;

            // Правая часть
            let psi_i = psi_old[i];
            let psi_l = if i > 0 {
                psi_old[i - 1]
            } else {
                C::new(0.0, 0.0)
            };
            let psi_r = if i < n - 1 {
                psi_old[i + 1]
            } else {
                C::new(0.0, 0.0)
            };

            let mut val = (C::new(1.0, 0.0) - i_dt_half * h_diag) * psi_i;
            if i > 0 {
                val -= i_dt_half * h_low * psi_l;
            }
            if i < n - 1 {
                val -= i_dt_half * h_up * psi_r;
            }
            *d_i = val;
        });

        let psi_new = self.solve_tridiagonal(&a_diag, &a_off_low, &a_off_up, &d);
        wf.psi.assign(&psi_new);
        t.current += t.dt;
    }

    /// Выполняет один шаг эволюции в мнимом времени (для нахождения основного состояния).
    ///
    /// Уравнение: $\displaystyle -\frac{\partial\psi}{\partial\tau} = \hat H\psi$.
    /// Схема Крэнка-Николсон:
    /// $$ \left(I + \frac{\Delta\tau}{2}\hat H\right)\psi^{\tau+\Delta\tau}
    ///    = \left(I - \frac{\Delta\tau}{2}\hat H\right)\psi^\tau. $$
    ///
    /// # Аргументы
    /// * `wf` — изменяемая волновая функция.
    /// * `t` — объект [`Tspace`], содержащий шаг `t.dt` (интерпретируется как $\Delta\tau$).
    ///   После выполнения шага `t.current` увеличивается на `t.dt`.
    /// * `potential` — замыкание для потенциала (зависит от комплексной координаты и времени,
    ///   хотя в мнимом времени потенциал обычно стационарен).
    ///
    /// # Примечание
    /// В мнимом времени норма не сохраняется, поэтому после каждого шага рекомендуется
    /// перенормировать волновую функцию (например, вызовом `wf.normalization_by_1()`).
    ///
    /// # Пример
    /// ```
    /// # use rssfm::dim1::space::Xspace1D;
    /// # use rssfm::dim1::wave_function::WaveFunction1D;
    /// # use rssfm::common::tspace::Tspace;
    /// # use rssfm::crank_nicolson::dim1::ecs_grid::ECSParams;
    /// # use rssfm::crank_nicolson::dim1::propagator::CrankNicolson1D;
    /// # let xspace = Xspace1D::new([-50.0], [0.2], [501]);
    /// # let params = ECSParams { r0: 8.0, theta: 0.0, width: 30.0 };
    /// # let mut propagator = CrankNicolson1D::new(&xspace, params);
    /// # let mut wf = WaveFunction1D::init_gauss_1d(xspace, 0.0, 0.0, 1.0);
    /// # propagator.init_wavefunction(&mut wf, |x| (2.0*std::f64::consts::PI).powf(-0.25)*(-x*x/4.0).exp());
    /// let mut t = Tspace::new(0.0, 0.1, 1, 1000);
    /// let potential = |x: Complex<f64>, _t: f64| -> Complex<f64> {
    ///     -Complex::new(1.0, 0.0) / (Complex::new(2.0, 0.0) + x * x).sqrt()
    /// };
    /// propagator.propagate_imaginary(&mut wf, &mut t, potential);
    /// wf.normalization_by_1();
    /// ```
    pub fn propagate_imaginary<F1>(
        &mut self,
        wf: &mut WaveFunction1D,
        t: &mut Tspace,
        potential: F1,
    ) where
        F1: Fn(C, F) -> C + Sync,
    {
        let n = self.grid.n_nodes();
        let psi_old = &wf.psi;
        let dtau_c = Complex::new(t.dt, 0.0);
        let half_dtau = dtau_c * 0.5;

        let mut d = Array1::<C>::zeros(n);
        let mut a_diag = Array1::<C>::zeros(n);
        let mut a_off_low = Array1::<C>::zeros(n);
        let mut a_off_up = Array1::<C>::zeros(n);

        multizip((
            d.iter_mut(),
            a_diag.iter_mut(),
            a_off_low.iter_mut(),
            a_off_up.iter_mut(),
        ))
        .enumerate()
        .par_bridge()
        .for_each(|(i, (d_i, a_diag_i, a_low_i, a_up_i))| {
            let x_i = self.grid.x[i];
            let v = potential(x_i, t.current);

            let alpha_i = self.alpha[i];
            let beta_i = self.beta[i];
            let gamma_i = self.gamma[i];

            let h_diag = beta_i + v;
            let h_low = alpha_i;
            let h_up = gamma_i;

            // A = I + dtau/2 * H
            *a_diag_i = Complex::new(1.0, 0.0) + half_dtau * h_diag;
            *a_low_i = half_dtau * h_low;
            *a_up_i = half_dtau * h_up;

            // d = (I - dtau/2 * H) * psi_old
            let psi_i = psi_old[i];
            let psi_l = if i > 0 {
                psi_old[i - 1]
            } else {
                Complex::new(0.0, 0.0)
            };
            let psi_r = if i < n - 1 {
                psi_old[i + 1]
            } else {
                Complex::new(0.0, 0.0)
            };

            let mut val = (Complex::new(1.0, 0.0) - half_dtau * h_diag) * psi_i;
            if i > 0 {
                val -= half_dtau * h_low * psi_l;
            }
            if i < n - 1 {
                val -= half_dtau * h_up * psi_r;
            }
            *d_i = val;
        });

        let psi_new = self.solve_tridiagonal(&a_diag, &a_off_low, &a_off_up, &d);
        wf.psi.assign(&psi_new);
        t.current += t.dt
    }

    /// Решает трёхдиагональную систему линейных уравнений методом прогонки (Thomas algorithm).
    ///
    /// Система имеет вид:
    /// $$ a\_low_i x_{i-1} + a\_diag_i x_i + a\_up_i x_{i+1} = d_i,\quad i=0,\ldots,N-1, $$
    /// с условиями $a\_low_0 = 0$ и $a\_up_{N-1} = 0$.
    ///
    /// # Аргументы
    /// * `a_diag` — главная диагональ.
    /// * `a_low` — нижняя диагональ (элемент `i` соответствует коэффициенту при $x_{i-1}$).
    /// * `a_up` — верхняя диагональ (элемент `i` соответствует коэффициенту при $x_{i+1}$).
    /// * `d` — правая часть.
    ///
    /// # Возвращаемое значение
    /// Вектор решения $x$.
    ///
    /// # Паника
    /// Функция паникует, если на каком-либо шаге знаменатель оказывается слишком малым
    /// (норма < $10^{-12}$), что свидетельствует о плохой обусловленности системы.
    fn solve_tridiagonal(
        &mut self,
        a_diag: &Array1<C>,
        a_low: &Array1<C>,
        a_up: &Array1<C>,
        d: &Array1<C>,
    ) -> Array1<C> {
        let n = a_diag.len();
        let c_prime = &mut self.c_prime;
        let d_prime = &mut self.d_prime;

        // Прямой ход
        if a_diag[0].norm() < 1e-12 {
            panic!("Zero diagonal at index 0");
        }
        c_prime[0] = a_up[0] / a_diag[0];
        d_prime[0] = d[0] / a_diag[0];

        for i in 1..n {
            let denom = a_diag[i] - a_low[i] * c_prime[i - 1];
            if denom.norm() < 1e-12 {
                println!("CRITICAL: denom near zero at i={}, value={:?}", i, denom);
                println!(
                    "a_diag[i]={:?}, a_low[i]={:?}, c_prime[i-1]={:?}",
                    a_diag[i],
                    a_low[i],
                    c_prime[i - 1]
                );
                panic!("Tridiagonal solver failed");
            }
            c_prime[i] = a_up[i] / denom;
            d_prime[i] = (d[i] - a_low[i] * d_prime[i - 1]) / denom;
        }

        // Обратный ход
        let mut x = Array1::<C>::zeros(n);
        x[n - 1] = d_prime[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i + 1];
        }
        x
    }
}

// use super::ecs_grid::{ECSGrid1D, ECSParams};
// use crate::common::tspace::Tspace;
// use crate::config::{C, F, I};
// use crate::dim1::space::Xspace1D;
// use crate::dim1::wave_function::WaveFunction1D;
// use itertools::multizip;
// use ndarray::prelude::*;
// use num_complex::Complex;
// use rayon::prelude::*;
//
// pub struct CrankNicolson1D {
//     pub grid: ECSGrid1D,
//     pub alpha: Array1<C>,
//     pub beta: Array1<C>,
//     pub gamma: Array1<C>,
//     pub c_prime: Array1<C>,
//     pub d_prime: Array1<C>,
// }
//
// impl CrankNicolson1D {
//     // pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
//     //     let grid = ECSGrid1D::new(x_grid, params);
//     //     let n = grid.n_nodes();
//     //     let ds = x_grid.dx[0];
//     //
//     //     let mut alpha = Array1::<C>::zeros(n);
//     //     let mut beta = Array1::<C>::zeros(n);
//     //     let mut gamma = Array1::<C>::zeros(n);
//     //
//     //     for i in 0..n {
//     //         let dxds = grid.dxds[i];
//     //         let d2xds2 = grid.d2xds2[i];
//     //         let dxds2 = dxds * dxds;
//     //         let dxds3 = dxds2 * dxds;
//     //
//     //         let inv_dxds2 = Complex::new(1.0, 0.0) / dxds2;
//     //         let term1 = -0.5 * inv_dxds2 / (ds * ds);
//     //         let term2 = d2xds2 / (4.0 * ds * dxds3);
//     //
//     //         alpha[i] = term1 - term2;
//     //         gamma[i] = term1 + term2;
//     //         beta[i] = inv_dxds2 / (ds * ds);
//     //     }
//     //
//     //     alpha[0] = Complex::new(0.0, 0.0);
//     //     gamma[n - 1] = Complex::new(0.0, 0.0);
//     //
//     //     Self {
//     //         grid,
//     //         alpha,
//     //         beta,
//     //         gamma,
//     //         c_prime: Array1::zeros(n),
//     //         d_prime: Array1::zeros(n),
//     //     }
//     // }
//     pub fn new(x_grid: &Xspace1D, params: ECSParams) -> Self {
//         let grid = ECSGrid1D::new(x_grid, params);
//         let n = grid.n_nodes();
//         // Шаг по параметрической координате (постоянный)
//         let ds = x_grid.dx[0];
//
//         let mut alpha = Array1::<C>::zeros(n);
//         let mut beta = Array1::<C>::zeros(n);
//         let mut gamma = Array1::<C>::zeros(n);
//
//         // Внутренние узлы (i = 1 .. n-2) используют центральную формулу на неравномерной сетке
//         for i in 1..n - 1 {
//             let x_prev = grid.x[i - 1];
//             let x_cur = grid.x[i];
//             let x_next = grid.x[i + 1];
//
//             let dx_minus = x_cur - x_prev; // комплексное
//             let dx_plus = x_next - x_cur; // комплексное
//             let dx_avg = x_next - x_prev; // комплексное
//
//             // Коэффициенты для второй производной d²/dx² (без множителя -1/2)
//             let a = C::new(2.0, 0.0) / (dx_avg * dx_minus);
//             let b = -C::new(2.0, 0.0) / dx_avg
//                 * (C::new(1.0, 0.0) / dx_plus + C::new(1.0, 0.0) / dx_minus);
//             let c = C::new(2.0, 0.0) / (dx_avg * dx_plus);
//
//             // Оператор кинетической энергии T = -1/2 d²/dx²
//             alpha[i] = -C::new(0.5, 0.0) * a;
//             beta[i] = -C::new(0.5, 0.0) * b;
//             gamma[i] = -C::new(0.5, 0.0) * c;
//         }
//
//         // Граничные узлы: условие Дирихле (psi = 0 на границах)
//         // Для i = 0 и i = n-1 обнуляем коэффициенты, чтобы в уравнении не было связи с внешними точками
//         // Можно также оставить односторонние разности, но для простоты оставим так.
//         // При этом в прогонке граничные значения будут вычислены, но они будут зависеть только от соседей.
//         alpha[0] = C::new(0.0, 0.0);
//         gamma[0] = C::new(0.0, 0.0); // для i=0 gamma не нужна, но оставим
//         alpha[n - 1] = C::new(0.0, 0.0);
//         gamma[n - 1] = C::new(0.0, 0.0);
//
//         Self {
//             grid,
//             alpha,
//             beta,
//             gamma,
//             c_prime: Array1::zeros(n),
//             d_prime: Array1::zeros(n),
//         }
//     }
//
//     /// Инициализирует волновую функцию:
//     /// - во внутренней области ($|s| \le R_0$) используется действительная координата `x.re`
//     ///   и задаётся через замыкание `init_fn`, принимающее `F`;
//     /// - во внешней области ($|s| > R_0$) значение устанавливается в ноль.
//     pub fn init_wavefunction(&self, wf: &mut WaveFunction1D, init_fn: impl Fn(F) -> C + Sync) {
//         multizip((wf.psi.iter_mut(), self.grid.s.iter(), self.grid.x.iter()))
//             .par_bridge()
//             .for_each(|(psi_i, &s_i, &x_i)| {
//                 if s_i.abs() <= self.grid.params.r0 {
//                     *psi_i = init_fn(x_i.re); // внутри R0 x действителен
//                 } else {
//                     *psi_i = Complex::new(0.0, 0.0);
//                 }
//             });
//     }
//
//     /// Вычисляет норму с учётом якобиана
//     pub fn compute_norm(&self, wf: &WaveFunction1D) -> F {
//         let ds = self.grid.s[1] - self.grid.s[0]; // шаг по параметрической координате
//         self.grid
//             .dxds
//             .iter()
//             .zip(wf.psi.iter())
//             .map(|(&dxds_i, &psi_i)| psi_i.norm_sqr() * dxds_i.norm())
//             .sum::<F>()
//             * ds
//     }
//
//     pub fn propagate<F1>(&mut self, wf: &mut WaveFunction1D, t: &mut Tspace, potential: F1)
//     where
//         F1: Fn(C, F) -> C + Sync,
//     {
//         let n = self.grid.n_nodes();
//         let psi_old = &wf.psi;
//         let dt_c = Complex::new(t.dt, 0.0);
//         let i_dt_half = I * dt_c * 0.5;
//
//         let mut d = Array1::<C>::zeros(n);
//         let mut a_diag = Array1::<C>::zeros(n);
//         let mut a_off_low = Array1::<C>::zeros(n);
//         let mut a_off_up = Array1::<C>::zeros(n);
//
//         multizip((
//             d.iter_mut(),
//             a_diag.iter_mut(),
//             a_off_low.iter_mut(),
//             a_off_up.iter_mut(),
//         ))
//         .enumerate()
//         .par_bridge()
//         .for_each(|(i, (d_i, a_diag_i, a_low_i, a_up_i))| {
//             let x_i = self.grid.x[i];
//             let v = potential(x_i, t.current);
//
//             let alpha_i = self.alpha[i];
//             let beta_i = self.beta[i];
//             let gamma_i = self.gamma[i];
//
//             let h_diag = beta_i + v;
//             let h_low = alpha_i;
//             let h_up = gamma_i;
//
//             *a_diag_i = Complex::new(1.0, 0.0) + i_dt_half * h_diag;
//             *a_low_i = i_dt_half * h_low;
//             *a_up_i = i_dt_half * h_up;
//
//             // Проверка на потенциальные проблемы
//             if a_diag_i.norm() < 1e-12 {
//                 println!("WARN: a_diag near zero at i={}, value={:?}", i, a_diag_i);
//             }
//
//             let psi_i = psi_old[i];
//             let psi_l = if i > 0 {
//                 psi_old[i - 1]
//             } else {
//                 Complex::new(0.0, 0.0)
//             };
//             let psi_r = if i < n - 1 {
//                 psi_old[i + 1]
//             } else {
//                 Complex::new(0.0, 0.0)
//             };
//
//             let mut val = (Complex::new(1.0, 0.0) - i_dt_half * h_diag) * psi_i;
//             if i > 0 {
//                 val -= i_dt_half * h_low * psi_l;
//             }
//             if i < n - 1 {
//                 val -= i_dt_half * h_up * psi_r;
//             }
//             *d_i = val;
//         });
//
//         let psi_new = self.solve_tridiagonal(&a_diag, &a_off_low, &a_off_up, &d);
//         wf.psi.assign(&psi_new);
//         t.current += t.dt
//     }
//     /// Выполняет один шаг эволюции в мнимом времени (для нахождения основного состояния).
//     /// Уравнение: d/dtau psi = -H psi, используется метод Крэнка-Николсон:
//     /// (I + dtau/2 * H) psi_new = (I - dtau/2 * H) psi_old
//     pub fn propagate_imaginary<F1>(
//         &mut self,
//         wf: &mut WaveFunction1D,
//         t: &mut Tspace,
//         potential: F1,
//     ) where
//         F1: Fn(C, F) -> C + Sync,
//     {
//         let n = self.grid.n_nodes();
//         let psi_old = &wf.psi;
//         let dtau_c = Complex::new(t.dt, 0.0);
//         let half_dtau = dtau_c * 0.5;
//
//         let mut d = Array1::<C>::zeros(n);
//         let mut a_diag = Array1::<C>::zeros(n);
//         let mut a_off_low = Array1::<C>::zeros(n);
//         let mut a_off_up = Array1::<C>::zeros(n);
//
//         multizip((
//             d.iter_mut(),
//             a_diag.iter_mut(),
//             a_off_low.iter_mut(),
//             a_off_up.iter_mut(),
//         ))
//         .enumerate()
//         .par_bridge()
//         .for_each(|(i, (d_i, a_diag_i, a_low_i, a_up_i))| {
//             let x_i = self.grid.x[i];
//             let v = potential(x_i, t.current);
//
//             let alpha_i = self.alpha[i];
//             let beta_i = self.beta[i];
//             let gamma_i = self.gamma[i];
//
//             let h_diag = beta_i + v;
//             let h_low = alpha_i;
//             let h_up = gamma_i;
//
//             // A = I + dtau/2 * H
//             *a_diag_i = Complex::new(1.0, 0.0) + half_dtau * h_diag;
//             *a_low_i = half_dtau * h_low;
//             *a_up_i = half_dtau * h_up;
//
//             // d = (I - dtau/2 * H) * psi_old
//             let psi_i = psi_old[i];
//             let psi_l = if i > 0 {
//                 psi_old[i - 1]
//             } else {
//                 Complex::new(0.0, 0.0)
//             };
//             let psi_r = if i < n - 1 {
//                 psi_old[i + 1]
//             } else {
//                 Complex::new(0.0, 0.0)
//             };
//
//             let mut val = (Complex::new(1.0, 0.0) - half_dtau * h_diag) * psi_i;
//             if i > 0 {
//                 val -= half_dtau * h_low * psi_l;
//             }
//             if i < n - 1 {
//                 val -= half_dtau * h_up * psi_r;
//             }
//             *d_i = val;
//         });
//
//         let psi_new = self.solve_tridiagonal(&a_diag, &a_off_low, &a_off_up, &d);
//         wf.psi.assign(&psi_new);
//         t.current += t.dt
//     }
//
//     fn solve_tridiagonal(
//         &mut self,
//         a_diag: &Array1<C>,
//         a_low: &Array1<C>,
//         a_up: &Array1<C>,
//         d: &Array1<C>,
//     ) -> Array1<C> {
//         let n = a_diag.len();
//         let c_prime = &mut self.c_prime;
//         let d_prime = &mut self.d_prime;
//
//         // Прямой ход
//         if a_diag[0].norm() < 1e-12 {
//             panic!("Zero diagonal at index 0");
//         }
//         c_prime[0] = a_up[0] / a_diag[0];
//         d_prime[0] = d[0] / a_diag[0];
//
//         for i in 1..n {
//             let denom = a_diag[i] - a_low[i] * c_prime[i - 1];
//             if denom.norm() < 1e-12 {
//                 println!("CRITICAL: denom near zero at i={}, value={:?}", i, denom);
//                 println!(
//                     "a_diag[i]={:?}, a_low[i]={:?}, c_prime[i-1]={:?}",
//                     a_diag[i],
//                     a_low[i],
//                     c_prime[i - 1]
//                 );
//                 panic!("Tridiagonal solver failed");
//             }
//             c_prime[i] = a_up[i] / denom;
//             d_prime[i] = (d[i] - a_low[i] * d_prime[i - 1]) / denom;
//         }
//
//         let mut x = Array1::<C>::zeros(n);
//         x[n - 1] = d_prime[n - 1];
//         for i in (0..n - 1).rev() {
//             x[i] = d_prime[i] - c_prime[i] * x[i + 1];
//         }
//         x
//     }
// }
