use num_complex::Complex;

// тип данных: f64 или f32
pub type F = f32;

// комплексный тип данных, согласованный с Float
pub type C = Complex<F>;

// константы
pub const PI: F = std::f32::consts::PI;
pub const I: C = Complex::I;
