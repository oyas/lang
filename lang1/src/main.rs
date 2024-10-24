use rustlang::run;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect::<Vec<String>>().split_off(1);

    let file_name = match args.iter().find(|&x| !x.starts_with("-")) {
        Some(file_name) => file_name,
        None => "",
    };

    let show_log = args.contains(&String::from("-v"));

    let result = run(file_name, show_log);

    match result {
        Some(el) => println!("{}", el.value.to_string()),
        None => {}
    }
}
