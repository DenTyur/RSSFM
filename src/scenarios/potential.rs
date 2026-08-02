//! Диспетчер атомных и поглощающих потенциалов по имени модели.
//! `model` (строка) — точное имя функции из `rssfm::potentials`, поэтому добавить
//! новый потенциал = дописать функцию в ядре + одна ветка в match здесь.
//! Принимают уже разрешённые значения; возвращают ОДНО замыкание (общий тип),
//! поэтому передаются в SSFM как `impl Fn` (Send + Sync), без trait-object.

use rssfm::config::{C, F};
use rssfm::potentials::absorbing_potentials::{
    absorbing_potential_1d, absorbing_potential_1d_asim, absorbing_potential_2d,
    absorbing_potential_2d_asim,
};
use rssfm::potentials::potentials;
use std::collections::HashMap;

fn p(params: &HashMap<String, f64>, name: &str) -> F {
    params
        .get(name)
        .copied()
        .unwrap_or_else(|| panic!("нет параметра {name:?} в [atomic potential].parameters")) as F
}

/// 2D атомный потенциал |[F;2]| -> F (одно замыкание для всех моделей).
pub fn atomic_2d(model: String, params: HashMap<String, f64>) -> impl Fn([F; 2]) -> F {
    move |x: [F; 2]| -> F {
        match model.as_str() {
            "soft_coulomb_2e1d_interact" => {
                potentials::soft_coulomb_2e1d_interact(x, p(&params, "z"), p(&params, "a"), p(&params, "b"))
            }
            "br_2e1d" => potentials::br_2e1d(x),
            "br_2e1d_com" => potentials::br_2e1d_com(x),
            other => panic!("неизвестный 2D атомный потенциал: {other}"),
        }
    }
}

/// 1D атомный потенциал |[F;1]| -> F.
pub fn atomic_1d(model: String, params: HashMap<String, f64>) -> impl Fn([F; 1]) -> F {
    move |x: [F; 1]| -> F {
        match model.as_str() {
            "soft_coulomb_1d" => potentials::soft_coulomb_1d(x, p(&params, "z"), p(&params, "a")),
            "br_1e1d_inner" => potentials::br_1e1d_inner(x),
            "br_1e1d_external" => potentials::br_1e1d_external(x),
            other => panic!("неизвестный 1D атомный потенциал: {other}"),
        }
    }
}

/// 2D поглощающий потенциал |[F;2]| -> C.
pub fn absorbing_2d(model: String, alpha: F, region: Vec<[f64; 2]>, radius: Option<f64>) -> impl Fn([F; 2]) -> C {
    move |x: [F; 2]| -> C {
        match model.as_str() {
            "absorbing_potential_2d_asim" => absorbing_potential_2d_asim(x, region_as_2d(&region), alpha),
            "absorbing_potential_2d" => {
                let r0: F = radius.expect("для absorbing_potential_2d нужен radius_absorber") as F;
                absorbing_potential_2d(x, r0, alpha)
            }
            other => panic!("неизвестный 2D поглощающий потенциал: {other}"),
        }
    }
}

/// 1D поглощающий потенциал |[F;1]| -> C.
pub fn absorbing_1d(model: String, alpha: F, region: Vec<[f64; 2]>, radius: Option<f64>) -> impl Fn([F; 1]) -> C {
    move |x: [F; 1]| -> C {
        match model.as_str() {
            "absorbing_potential_1d_asim" => {
                let r0: [F; 2] = [region[0][0] as F, region[0][1] as F];
                absorbing_potential_1d_asim(x, r0, alpha)
            }
            "absorbing_potential_1d" => {
                let r0: F = radius.expect("для absorbing_potential_1d нужен radius_absorber") as F;
                absorbing_potential_1d(x, r0, alpha)
            }
            other => panic!("неизвестный 1D поглощающий потенциал: {other}"),
        }
    }
}

fn region_as_2d(region: &[[f64; 2]]) -> [[F; 2]; 2] {
    if region.len() != 2 {
        panic!("для 2D asim нужен регион на две оси в [absorbing potential].region");
    }
    [[region[0][0] as F, region[0][1] as F], [region[1][0] as F, region[1][1] as F]]
}