//! Локальный макрос логирования для параметризованных сценариев.
//! Пишет и в stdout, и в файл <output.dir>/output.log (в отличие от ядерного
//! `print_and_log!`, который пишет в output.log текущего каталога).

/// Замер времени: печать в stdout + запись длительности в <dir>/output.log.
#[macro_export]
macro_rules! scenario_measure_time {
    ($log_path:expr, $label:expr, $code:block) => {
        let start = std::time::Instant::now();
        $code
        let duration = start.elapsed().as_secs_f32();
        println!("{}: {:.3}", $label, duration);
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            if let Some(p) = std::path::Path::new($log_path).parent() {
                let _ = std::fs::create_dir_all(p);
            }
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open($log_path) {
                writeln!(file, "{}: {:.3}", $label, duration)
                    .unwrap_or_else(|e| eprintln!("Failed to write to log file: {}", e));
            } else {
                eprintln!("Failed to open log file");
            }
        }
    };
}

/// Печать в stdout + дописывание в файл по указанному пути.
#[macro_export]
macro_rules! scenario_log {
    ($log_path:expr, $($arg:tt)*) => {{
        println!($($arg)*);
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            if let Some(p) = std::path::Path::new($log_path).parent() {
                let _ = std::fs::create_dir_all(p);
            }
            match OpenOptions::new().create(true).append(true).open($log_path) {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, $($arg)*) {
                        eprintln!("Failed to write to log file: {}", e);
                    }
                }
                Err(e) => eprintln!("Failed to open log file: {}", e),
            }
        }
    }};
}