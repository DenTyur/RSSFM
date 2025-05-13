/// # Математические формулы
///
/// <!-- MathJax для рендеринга формул -->
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
///
/// Формула площади круга:  
/// \\[ S = \pi r^2 \\]
pub mod config;
pub mod evolution;
pub mod field;
pub mod flow;
pub mod gauge;
pub mod heatmap;
pub mod logcolormap;
pub mod macros;
pub mod parameters;
pub mod pml;
pub mod potentials;
pub mod tsurff;
pub mod volkov;
pub mod wave_function;

#[cfg(test)]
mod tests;
