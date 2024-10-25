mod ast;
mod backend;
mod parser;
mod compiler;

use std::env;

fn main() {
    // println!("Hello, world!");
    // let ret = backend::llvm::compile();
    // println!("{:?}", ret);
    let args: Vec<String> = env::args().collect::<Vec<String>>().split_off(1);

    let file_name = match args.iter().find(|&x| !x.starts_with("-")) {
        Some(file_name) => file_name,
        None => "",
    };

    // let show_log = args.contains(&String::from("-v"));

    if !file_name.is_empty() {
        println!("compile {}", file_name);
        compiler::compile(&file_name);
        return;
    }

    compiler::repl();
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn test() {
        assert_eq!(4, 4);
    }
}