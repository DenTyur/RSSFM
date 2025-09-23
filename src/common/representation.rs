/// Перечисление для указания, вкаком представлении находитсяволновая функция
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub enum Representation {
    Position, // координатное представление
    Momentum, // импульсное представление
}

impl Representation {
    pub fn as_str(&self) -> &str {
        match self {
            Representation::Position => "Position",
            Representation::Momentum => "Momentum",
        }
    }
}
