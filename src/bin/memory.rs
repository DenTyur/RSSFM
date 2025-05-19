use std::io;

fn main() {
    println!("Размерность:");
    let mut dim = String::new();
    io::stdin()
        .read_line(&mut dim)
        .expect("Не удалось прочитать строку");

    let dim: u32 = dim.trim().parse().expect("Ожидалось u32");

    println!("Число точек на каждой из осей:");
    let mut n = String::new();
    io::stdin()
        .read_line(&mut n)
        .expect("Не удалось прочитать строку");

    let n: i64 = n.trim().parse().expect("Ожидалось i64");
    let bytes_to_gb: i64 = 2_i64.pow(30);
    let c32: i64 = 8;
    // let n: i64 = 160;
    let wf = n.pow(dim) * c32;
    let temp_wf = wf.clone();
    let xspace = dim as i64 * n * c32;
    let pspace = xspace.clone();
    let memory_ssfm: f64 = (wf + temp_wf + xspace + pspace) as f64 / bytes_to_gb as f64;
    println!("memory_ssfm = {} Gb", memory_ssfm);
}
