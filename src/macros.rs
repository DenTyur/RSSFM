/// Макрос для замера времени
#[macro_export]
macro_rules! measure_time {
    ($label:expr, $code:block) => {
        let start = std::time::Instant::now();
        $code
        let duration = start.elapsed().as_secs_f32();
        println!("{}: {:.3}", $label, duration);
    };
}

/// Макрос для проверки существования директории и создания ее
macro_rules! check_path {
    ($path:expr) => {
        use std::fs;
        use std::path::Path;
        let path = Path::new($path);
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .expect(&format!("Failed to create directory: {:?}", parent));
            }
        }
    };
}
pub(crate) use check_path;
