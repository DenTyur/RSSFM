// use gsl::sf::bessel;
// use num_complex::Complex64;
// use std::f64::consts::{PI, SQRT_2};
//
// /// Вычисляет 2D кулоновскую волновую функцию ψ_p(r) для заданного импульса и координаты.
// ///
// /// # Аргументы
// /// * `p` - Импульс [p_x, p_y] (в единицах ħ).
// /// * `r` - Координата [x, y].
// /// * `charge` - Параметр Z e^2.
// /// * `mass` - Масса частицы (в единицах ħ = 1).
// ///
// /// # Возвращает
// /// Комплексное значение ψ_p(r).
// #[test]
// fn coulomb_wave_2d(p: [f64; 2], r: [f64; 2], charge: f64, mass: f64) -> Complex64 {
//     let k = (p[0].powi(2) + p[1].powi(2)).sqrt(); // |k| = p / ħ
//     let eta = charge * mass / k; // η = Z e^2 m / (ħ k)
//     let r_norm = (r[0].powi(2) + r[1].powi(2)).sqrt(); // r = √(x² + y²)
//     let kr = k * r_norm;
//     let phi_p = p[1].atan2(p[0]); // Угол импульса
//     let phi_r = r[1].atan2(r[0]); // Угол координаты
//
//     // Нормировочная константа
//     let c = (PI * eta / 2.0).sinh().sqrt();
//
//     // Разложение по угловым моментам (m ∈ ℤ)
//     let mut psi = Complex64::new(0.0, 0.0);
//     for m in -20..=20 {
//         let sigma_m = (gamma(1 + m + eta * Complex64::i()).arg(); // Фазовый сдвиг
//         let j_m = bessel::Jnu(m as f64, kr); // Функция Бесселя
//         let y_m = bessel::Ynu(m as f64, kr); // Функция Неймана
//
//         psi += Complex64::i().powi(m)
//             * Complex64::from_polar(1.0, sigma_m)
//             * (j_m + Complex64::i() * y_m)
//             * Complex64::from_polar(1.0, m as f64 * (phi_r - phi_p));
//     }
//
//     psi * c * (-PI * eta / 2.0).exp()
// }
