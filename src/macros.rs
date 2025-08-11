/// Макрос для замера времени
#[macro_export]
macro_rules! measure_time {
    ($label:expr, $code:block) => {
        let start = std::time::Instant::now();
        $code
        let duration = start.elapsed().as_secs_f32();
        println!("{}: {:.3}", $label, duration);
        // Запись в файл
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open("output.log")
            {
                writeln!(file, "{}: {:.3}", $label, duration).unwrap_or_else(|e| eprintln!("Failed to write to log file: {}", e));
            } else {
                eprintln!("Failed to open log file");
            }
        }
    };
}

/// Макрос для вывода в терминал и в файл
#[macro_export]
macro_rules! print_and_log {
    ($($arg:tt)*) => {{
        // Вывод в терминал
        println!($($arg)*);

        // Запись в файл
        use std::fs::OpenOptions;
        use std::io::Write;
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open("output.log")
        {
            Ok(mut file) => {
                if let Err(e) = writeln!(file, $($arg)*) {
                    eprintln!("Failed to write to log file: {}", e);
                }
            },
            Err(e) => eprintln!("Failed to open log file: {}", e),
        }
    }};
}

/// Макрос для проверки существования директории и создания ее
macro_rules! check_path {
    ($path:expr) => {
        let path = std::path::Path::new($path);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .expect(&format!("Failed to create directory: {:?}", parent));
            }
        }
    };
}
pub(crate) use check_path;
