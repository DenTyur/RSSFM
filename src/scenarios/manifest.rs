//! Запись manifest в выходной каталог расчёта: параметры, git-хэш, дата.
//! Это делает расчёт «перезапускаемым» спустя время.

use super::config::Config;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Возвращает текущий git commit (rev-parse HEAD), если доступен.
pub fn git_head() -> Option<String> {
    let out = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if out.status.success() {
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        None
    }
}

/// Возвращает краткий статус git (git describe / статус рабочей копии).
pub fn git_status() -> String {
    match Command::new("git").args(["status", "--porcelain"]).output() {
        Ok(out) if out.status.success() => {
            let clean_flag = if out.stdout.is_empty() { "clean" } else { "dirty" };
            format!("{clean_flag} ({} file(s) changed)", count_lines(&out.stdout))
        }
        _ => "git status unavailable".to_string(),
    }
}

fn count_lines(b: &[u8]) -> usize {
    String::from_utf8_lossy(b).lines().filter(|l| !l.is_empty()).count()
}

/// Записать manifest.txt в каталог out_dir.
pub fn write_manifest(cfg: &Config, out_dir: &str) {
    std::fs::create_dir_all(out_dir).ok();
    let mut text = String::new();
    match toml::to_string_pretty(cfg) {
        Ok(s) => text.push_str(&s),
        Err(e) => text.push_str(&format!("(не удалось сериализовать конфиг: {e})\n")),
    }
    text.push_str("\n[reproducibility]\n");
    text.push_str(&format!("git_head = {:?}\n", git_head().unwrap_or_else(|| "(нет git)".into())));
    text.push_str(&format!("git_status = {:?}\n", git_status()));
    text.push_str(&format!(
        "written_at_unix = {}\n",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    let path = format!("{out_dir}/manifest.txt");
    std::fs::write(&path, text).expect("не удалось записать manifest");
    println!("manifest записан в {path}");
}