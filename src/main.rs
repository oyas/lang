mod ast;
mod backend;
mod parser;
mod compiler;

use std::env;

use compiler::Options;

fn main() {
    // println!("Hello, world!");
    // let ret = backend::llvm::compile();
    // println!("{:?}", ret);
    let args: Vec<String> = env::args().collect::<Vec<String>>().split_off(1);

    let file_name = match args.iter().find(|&x| !x.starts_with("-")) {
        Some(file_name) => file_name,
        None => "",
    };

    let options = Options {
        emit_llvm_ir: args.contains(&String::from("--emit-llvm-ir")),
        emit_ast: args.contains(&String::from("--emit-ast")),
    };

    if !file_name.is_empty() {
        println!("compile {}", file_name);
        compiler::compile(&file_name, &options);
        return;
    }

    compiler::repl(&options);
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn test() {
        assert_eq!(4, 4);
    }
}