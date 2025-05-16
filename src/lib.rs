pub mod common;
pub mod config;
pub mod macros;
pub mod potentials;
#[cfg(test)]
mod tests;
pub mod utils;

// #[cfg(feature = "dim2")]
pub mod dim2;
// #[cfg(feature = "dim2")]
// pub use dim2::*;

// #[cfg(feature = "dim4")]
pub mod dim4;
// #[cfg(feature = "dim4")]
// pub use dim4::*;

pub mod traits;
