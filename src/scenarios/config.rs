use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Visitor};
use std::collections::HashMap;
use std::str::FromStr;

// =====================================================================================
//                                Механизм выражений
// =====================================================================================
// Любое числовое поле может быть задано числом ЛИБО строкой-выражением (meval).
// Переменные в выражении разрешаются из контекста = секции [constants] + число-соседи.
// Пример: dt = "0.1 / au_to_fs / n_steps"
//   [constants]
//   au_to_fs = 2.4188843e-2
// Имена полей в toml повторяют имена соответствующих структур ядра (сооб.)
// =====================================================================================

#[derive(Debug, Clone)]
pub enum NumOrExpr {
    Num(f64),
    Str(String),
}

impl NumOrExpr {
    /// Вычислить значение из контекста (имена -> числа). Число возвращается как есть.
    pub fn eval(&self, ctx: &HashMap<String, f64>) -> f64 {
        match self {
            NumOrExpr::Num(v) => *v,
            NumOrExpr::Str(s) => {
                let expr = meval::Expr::from_str(s)
                    .unwrap_or_else(|_| panic!("не удалось разобрать выражение: {s:?}"));
                let mut c = meval::Context::new();
                c.var("PI", rssfm::config::PI as f64);
                c.var("pi", rssfm::config::PI as f64);
                for (name, val) in ctx {
                    c.var(name.clone(), *val);
                }
                expr.eval_with_context(&c)
                    .unwrap_or_else(|e| panic!("ошибка вычисления выражения {s:?}: {e}"))
            }
        }
    }
}

struct NumOrVisitor;

impl<'de> Visitor<'de> for NumOrVisitor {
    type Value = NumOrExpr;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "число или строка-выражение")
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<NumOrExpr, E> {
        Ok(NumOrExpr::Num(v))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<NumOrExpr, E> {
        Ok(NumOrExpr::Num(v as f64))
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<NumOrExpr, E> {
        Ok(NumOrExpr::Num(v as f64))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<NumOrExpr, E> {
        Ok(NumOrExpr::Str(v.to_string()))
    }
}

impl Serialize for NumOrExpr {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            NumOrExpr::Num(v) => s.serialize_f64(*v),
            NumOrExpr::Str(st) => s.serialize_str(st),
        }
    }
}

impl<'de> Deserialize<'de> for NumOrExpr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NumOrVisitor)
    }
}

// =====================================================================================
//                                   Разделы конфига
// =====================================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Константы, доступные в выражениях (имена = переменные)
    #[serde(default)]
    pub constants: HashMap<String, f64>,
    #[serde(default)]
    pub grid: Option<GridCfg>,
    #[serde(default)]
    pub time: Option<TimeCfg>,
    #[serde(rename = "external field", default)]
    pub external_field: Option<FieldCfg>,
    #[serde(rename = "atomic potential", default)]
    pub atomic_potential: Option<AtomicCfg>,
    #[serde(rename = "absorbing potential", default)]
    pub absorbing_potential: Option<AbsCfg>,
    #[serde(rename = "initial state", default)]
    pub initial_state: Option<InitCfg>,
    #[serde(rename = "ionization probabilities r_surfaces", default)]
    pub ioniz: Option<IonCfg>,
    #[serde(rename = "ionization probabilities x_surfaces", default)]
    pub ioniz_x: Option<IonXSurfCfg>,
    #[serde(rename = "survival probability", default)]
    pub survival: Option<SurvivalCfg>,
    #[serde(rename = "time fft", default)]
    pub time_fft: Option<TimeFftCfg>,
    pub output: Option<OutputCfg>,
    #[serde(default)]
    pub particles: Option<Vec<ParticleCfg>>,
    #[serde(rename = "sym noninteracting state", default)]
    pub sym_nonint: Option<SymCfg>,
}

// --- [grid] : соответствует Xspace1D/2D {x0, dx, n} ----------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridCfg {
    pub x0: Vec<NumOrExpr>,
    pub dx: Vec<NumOrExpr>,
    pub n: Vec<usize>,
}

// --- [time] : соответствует Tspace {t0, dt, n_steps, nt} ------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeCfg {
    pub t0: NumOrExpr,
    pub dt: NumOrExpr,
    pub n_steps: usize,
    pub nt: usize,
}

// --- ["external field"] -> поля UnipolarPulse {amplitude, omega, x_envelop} -----------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldCfg {
    /// имя типа поля, например "UnipolarPulse2e1d" / "UnipolarPulse1D"
    pub pulse_shape: String,
    pub amplitude: NumOrExpr,
    pub omega: NumOrExpr,
    pub x_envelop: NumOrExpr,
}

// --- ["atomic potential"] --------------------------------------------------------------
// model — точное имя функции в rssfm::potentials; parameters — её аргументы без имён
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicCfg {
    pub model: String,
    #[serde(default)]
    pub parameters: HashMap<String, NumOrExpr>,
}

// --- ["absorbing potential"] -----------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsCfg {
    pub model: String,
    /// границы по каждой оси: [[x_min,x_max],[y_min,y_max]] (1D -> [[x_min,x_max]])
    pub region: Vec<[NumOrExpr; 2]>,
    pub alpha: NumOrExpr,
    #[serde(default)]
    pub radius_absorber: Option<NumOrExpr>,
}

// --- ["initial state"] -----------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitCfg {
    pub hdf5_file: String,
    #[serde(default = "default_true")]
    pub extend_to_grid: bool,
    #[serde(default = "default_true")]
    pub renormalize: bool,
}

// --- ["ionization probabilities r_surfaces"] -> поле r_surf в DoubleIonizProb2e1d -------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonCfg {
    /// r_surf: Vec<F> — радиусы поверхностей (имя как в структуре DoubleIonizProb2e1d)
    pub r_surf: Vec<NumOrExpr>,
}

// --- ["ionization probabilities x_surfaces"] -> поле x_surf в Ion1zProb1D ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonXSurfCfg {
    /// x_surf: Vec<F> — координаты поверхностей (имя как в структуре IonizProb1D)
    pub x_surf: Vec<NumOrExpr>,
}

// --- ["survival probability"] -> отслеживание |<psi0|psi(t)>|^2 (ProjectionProb1D) ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalCfg {
    #[serde(default = "default_true")]
    pub save_plot: bool,
    #[serde(default = "default_true")]
    pub save_data: bool,
}

// --- ["time fft"]: TimeFFT (временное Фурье-преобразование для основного состояния) ---
// Дополнительная секция; включается присутствием самой секции.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFftCfg {
    /// координаты точки (для 1D — один элемент, для 2D — два), как в TimeFFT::new
    pub point: Vec<NumOrExpr>,
    #[serde(default = "default_true")]
    pub save_plot: bool,
    #[serde(default = "default_true")]
    pub save_data: bool,
    /// границы по энергии для plot_log (1D; необязательно, по умолчанию [-40, 40])
    #[serde(default)]
    pub energy_limits: Option<Vec<NumOrExpr>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCfg {
    pub directory: String,
    #[serde(default = "default_true")]
    pub save_time_series_grid: bool,
    #[serde(default = "default_true")]
    pub save_wavefunctions_hdf5: bool,
    #[serde(default = "default_true")]
    pub save_plots_png: bool,
}

// --- [[particles]] -> Particle {dim, mass, charge} -------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleCfg {
    pub dim: usize,
    pub mass: NumOrExpr,
    pub charge: NumOrExpr,
}

// --- ["sym noninteracting state"] ------------------------------------------------------
// Сетка берётся из .hdf5 волновых функций (поле WFun.x), inner_grid_root НЕ нужен.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymCfg {
    /// каталог с волнами внутреннего электрона (папка psi_x)
    pub inner_wavefunction_root: String,
    /// каталог с волнами внешнего электрона (папка psi_x)
    pub external_wavefunction_root: String,
    /// путь к t.npy (столбик моментов-сесий)
    pub time_grid_path: String,
}

fn default_true() -> bool {
    true
}

// =====================================================================================
//                        Разрешённые (финальные) величины для раннеров
// =====================================================================================

/// Аргументы Tspace: t0, dt, n_steps, nt
#[derive(Debug, Clone)]
pub struct TimeRes {
    pub t0: f64,
    pub dt: f64,
    pub n_steps: usize,
    pub nt: usize,
}

#[derive(Debug, Clone)]
pub struct GridRes {
    pub x0: Vec<f64>,
    pub dx: Vec<f64>,
    pub n: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct FieldRes {
    pub pulse_shape: String,
    pub amplitude: f64,
    pub omega: f64,
    pub x_envelop: f64,
}

#[derive(Debug, Clone)]
pub struct AbsRes {
    pub model: String,
    pub region: Vec<[f64; 2]>,
    pub alpha: f64,
    pub radius: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TimeFftRes {
    pub point: Vec<f64>,
    pub save_plot: bool,
    pub save_data: bool,
    pub energy_limits: Option<Vec<f64>>,
}

impl Config {
    pub fn load(path: &str) -> Result<Config, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("не удалось прочитать {path}: {e}"))?;
        toml::from_str(&text).map_err(|e| format!("ошибка парсинга {path}: {e}"))
    }

    fn need<T>(opt: Option<T>, name: &str) -> T {
        opt.unwrap_or_else(|| panic!("в конфиге нет секции {name}"))
    }

    pub fn resolve_time(&self) -> TimeRes {
        let t = Self::need(self.time.as_ref(), "[time]");
        let mut ctx = self.constants.clone();
        ctx.insert("n_steps".into(), t.n_steps as f64);
        ctx.insert("nt".into(), t.nt as f64);
        TimeRes {
            t0: t.t0.eval(&ctx),
            dt: t.dt.eval(&ctx),
            n_steps: t.n_steps,
            nt: t.nt,
        }
    }

    pub fn resolve_grid(&self) -> GridRes {
        let g = Self::need(self.grid.as_ref(), "[grid]");
        let ctx = &self.constants;
        GridRes {
            x0: g.x0.iter().map(|v| v.eval(ctx)).collect(),
            dx: g.dx.iter().map(|v| v.eval(ctx)).collect(),
            n: g.n.clone(),
        }
    }

    pub fn resolve_field(&self) -> FieldRes {
        let f = Self::need(self.external_field.as_ref(), "[\"external field\"]");
        let ctx = &self.constants;
        FieldRes {
            pulse_shape: f.pulse_shape.clone(),
            amplitude: f.amplitude.eval(ctx),
            omega: f.omega.eval(ctx),
            x_envelop: f.x_envelop.eval(ctx),
        }
    }

    pub fn resolve_parameters(&self) -> HashMap<String, f64> {
        let a = Self::need(self.atomic_potential.as_ref(), "[\"atomic potential\"]");
        let ctx = &self.constants;
        a.parameters.iter().map(|(k, v)| (k.clone(), v.eval(ctx))).collect()
    }

    pub fn atomic_model(&self) -> String {
        Self::need(self.atomic_potential.as_ref(), "[\"atomic potential\"]")
            .model
            .clone()
    }

    /// Поглощающий потенциал. Возвращает None, если секция ["absorbing potential"] отсутствует.
    pub fn resolve_abs(&self) -> Option<AbsRes> {
        let a = self.absorbing_potential.as_ref()?;
        let ctx = &self.constants;
        Some(AbsRes {
            model: a.model.clone(),
            region: a.region.iter().map(|r| [r[0].eval(ctx), r[1].eval(ctx)]).collect(),
            alpha: a.alpha.eval(ctx),
            radius: a.radius_absorber.as_ref().map(|v| v.eval(ctx)),
        })
    }

    pub fn resolve_r_surf(&self) -> Vec<f64> {
        let ion = Self::need(self.ioniz.as_ref(), "[\"ionization probabilities r_surfaces\"]");
        let ctx = &self.constants;
        ion.r_surf.iter().map(|v| v.eval(ctx)).collect()
    }

    pub fn initial_root(&self) -> String {
        Self::need(self.initial_state.as_ref(), "[\"initial state\"]").hdf5_file.clone()
    }

    /// Вычислить x_surf для `["ionization probabilities x_surfaces"]` (1D).
    /// Возвращает None, если секции нет (функция опциональна).
    pub fn resolve_x_surf(&self) -> Option<Vec<f64>> {
        self.ioniz_x.as_ref().map(|ion| {
            let ctx = &self.constants;
            ion.x_surf.iter().map(|v| v.eval(ctx)).collect()
        })
    }

    /// Данные для TimeFFT. Возвращает None, если секции ["time fft"] нет.
    pub fn time_fft(&self) -> Option<TimeFftRes> {
        self.time_fft.as_ref().map(|f| {
            let ctx = &self.constants;
            TimeFftRes {
                point: f.point.iter().map(|v| v.eval(ctx)).collect(),
                save_plot: f.save_plot,
                save_data: f.save_data,
                energy_limits: f.energy_limits.as_ref().map(|v| v.iter().map(|e| e.eval(ctx)).collect()),
            }
        })
    }

    pub fn init_flags(&self) -> (bool, bool) {
        let i = Self::need(self.initial_state.as_ref(), "[\"initial state\"]");
        (i.extend_to_grid, i.renormalize)
    }

    pub fn output(&self) -> OutputCfg {
        Self::need(self.output.clone(), "[output]")
    }

    pub fn resolve_particles(&self) -> Vec<(usize, f64, f64)> {
        let ctx = &self.constants;
        self.particles
            .as_ref()
            .unwrap_or(&vec![])
            .iter()
            .map(|p| (p.dim, p.mass.eval(ctx), p.charge.eval(ctx)))
            .collect()
    }
}