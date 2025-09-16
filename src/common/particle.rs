use crate::config::F;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub dim: usize,
    pub mass: F,
    pub charge: F,
}
