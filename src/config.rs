use num_complex::Complex;

// тип данных: f64 или f32
pub type Float = f32;

// комплексный тип данных, согласованный с Float
pub type Compl = Complex<Float>;

// константы
pub const PI: Float = std::f32::consts::PI;
pub const I: Compl = Complex::I;
