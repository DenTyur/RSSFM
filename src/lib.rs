pub mod common;
pub mod config;
pub mod imaginary_time_evolution;
pub mod macros;
pub mod potentials;
pub mod utils;
pub use config::{C, F};

pub mod dim1;

// #[cfg(feature = "dim2")]
pub mod dim2;
// #[cfg(feature = "dim2")]
// pub use dim2::*;

pub mod dim3;

// #[cfg(feature = "dim4")]
pub mod dim4;
// #[cfg(feature = "dim4")]
// pub use dim4::*;

pub mod traits;
