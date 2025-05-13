/// Макрос для замера времени
macro_rules! measure_time {
    ($label:expr, $code:block) => {
        let start = std::time::Instant::now();
        $code
        let duration = start.elapsed().as_secs_f32();
        println!("{}: {:.3}", $label, duration);
    };
}
pub(crate) use measure_time;
