mod ast;
mod backend;
mod parser;
mod compiler;
mod hir;

use std::env;

use compiler::Options;

fn main() {
    let args: Vec<String> = env::args().collect::<Vec<String>>().split_off(1);

    let file_name = match args.iter().find(|&x| !x.starts_with("-")) {
        Some(file_name) => file_name,
        None => "",
    };

    let options = Options {
        emit_llvm_ir: args.contains(&String::from("--emit-llvm-ir")),
        emit_mlir: args.contains(&String::from("--emit-mlir")),
        emit_hir: args.contains(&String::from("--emit-hir")),
        emit_ast: args.contains(&String::from("--emit-ast")),
    };

    if !file_name.is_empty() {
        println!("compile {}", file_name);
        compiler::compile(&file_name, &options);
        return;
    }

    // compiler::repl(&options);
    compiler::repl_mlir::repl(&options);
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn test() {
        assert_eq!(4, 4);
    }
}