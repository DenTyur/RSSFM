use std::io;

fn main() {
    println!("Потенциал ионизации в эВ:");
    let mut ip = String::new();
    io::stdin()
        .read_line(&mut ip)
        .expect("Не удалось прочитать строку");

    let ip: f64 = ip.trim().parse().expect("Ожидалось u32");

    println!("E0: ");
    let mut E0 = String::new();
    io::stdin()
        .read_line(&mut E0)
        .expect("Не удалось прочитать строку");

    let E0: f64 = E0.trim().parse().expect("Ожидалось i64");

    println!("omega: ");
    let mut omega = String::new();
    io::stdin()
        .read_line(&mut omega)
        .expect("Не удалось прочитать строку");

    let omega: f64 = omega.trim().parse().expect("Ожидалось i64");

    let iat = 4.3597e-11; // атомная единица энергии в СГС

    let ev_to_erg = 1.60217733e-12;
    let ev_to_au = ev_to_erg / iat;
    let gamma = omega * (2.0 * ip * ev_to_au).sqrt() / E0;
    println!("gamma = {}", gamma);
}
