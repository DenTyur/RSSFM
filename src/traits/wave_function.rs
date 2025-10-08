use crate::config::{C, F};
use ndarray_npy::WriteNpyError;

/// Трейт для вычисления зачения и производной волновой функции.
///
/// Производную и значение могут иметь не только структура типа
/// `WaveFunction`, но и аналитически заданные волновые функции,
/// например, волковские функции, гауссов пакет и т.п.
///
/// Для расчета потоков, проекций и t-surff вычисление значения функции
/// и ее производной в точке вынесен в отдельный трейт.
///
/// D задает размерность пространства.
pub trait ValueAndSpaceDerivatives<const D: usize> {
    /// Возвращает производную функции вдоль оси axis: 0..D в точке x
    fn deriv(&self, x: [F; D], axis: usize) -> C;

    /// Возвращает значение функции в точке
    fn value(&self, x: [F; D]) -> C;
}

/// Трейт для волновой функции.
/// D задает размерность пространства.
pub trait WaveFunction<const D: usize> {
    type Xspace;

    // ==============Действия в волновой функцией================
    /// Расширяет сетку волновой функции нулями или обрезает сетку
    fn extend(&mut self, x_new: &Self::Xspace);

    /// Вычисляет и обновляет производные
    fn update_derivatives(&mut self);

    /// Возвращает вероятность (квадрат нормы) во всей расчетной области
    fn prob_in_numerical_box(&self) -> F;

    /// Возвращает норму волновой функции во всей расчетной области
    fn norm(&self) -> F;

    /// Нормирует волновую функцию на 1
    fn normalization_by_1(&mut self);

    /// Разрежает волновую функцию с шагом
    fn sparse(&mut self, sparse_step: isize);

    // ================== Сохранение в файл ======================
    /// Сохраняет волновую функцию в файл в формате .npy
    fn save_as_npy(&self, path: &str) -> Result<(), WriteNpyError>;

    /// Сохраняет волновую функцию в файл в формате .hdf5
    fn save_as_hdf5(&self, path: &str);

    /// Сохраняет волновую функцию в файл в формате .npy разрежая ее
    fn save_sparsed_as_npy(&self, path: &str, sparse_step: isize) -> Result<(), WriteNpyError>;

    /// Сохраняет разреженную волновую функцию в формате .hdf5
    fn save_sparsed_as_hdf5(&self, path: &str, sparse_step: isize);

    // ================== Считывание из файла ====================

    /// Считывает волновую функцию из .hdf5
    fn init_from_hdf5(psi_path: &str) -> Self;

    // Считывает волновую функцию из файла .npy
    fn init_from_npy(psi_path: &str, x: Self::Xspace) -> Self;
}
