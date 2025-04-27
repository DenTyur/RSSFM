use crate::field;
use field::Field2D;

#[derive(Clone, Copy)]
pub struct VelocityGauge<'a> {
    // без A^2
    pub field: &'a Field2D,
}

impl<'a> VelocityGauge<'a> {
    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}

#[derive(Clone, Copy)]
pub struct LenthGauge<'a> {
    pub field: &'a Field2D,
}

impl<'a> LenthGauge<'a> {
    pub fn new(field: &'a Field2D) -> Self {
        Self { field }
    }
}
