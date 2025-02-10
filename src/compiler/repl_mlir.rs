use std::io::Write;

use inkwell::context::Context;

use crate::{ast::Ast, backend::{self, mlir::CodeGen}, compiler::hir_to_mlir, hir::converter::convert_from_ast, parser::{self, is_complete}};

use super::Options;


pub fn repl(options: &Options) {
    let inkwell_context = Context::create();
    let mut inkwell_codegen = backend::llvm::CodeGen::new(&inkwell_context);
    let mut codegen = CodeGen::new();
    loop {
        print!("REPL> ");
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        loop {
            let mut s = String::new();
            std::io::stdin().read_line(&mut s).ok();
            input += &s;
            if is_complete(&input).is_ok() {
                break;
            }
        }
        if input == "" {
            println!("quit");
            break;
        }
        let ast = match parser::parse(&input) {
            Ok((_, r)) => r,
            Err(e) => {
                println!("parse error! {:?}", e);
                continue;
            }
        };
        if options.emit_ast {
            println!("parsed: {:?}", ast);
        }
        eval(&mut codegen, &inkwell_context, &mut inkwell_codegen, &ast, options);
    }
}

pub fn eval<'a>(
    codegen: &mut CodeGen<'a>,
    inkwell_context: &'a Context,
    inkwell_codegen: &mut backend::llvm::CodeGen<'a>,
    ast: &Ast,
    options: &Options
) {
    // Name of eval module
    let num = codegen.modules.len();
    let module_name = format!("__eval_{}", num);

    // Convert AST to HIR
    let hir = convert_from_ast(ast, &module_name);
    if options.emit_hir {
        println!("HIR: {:?}", hir);
    }

    // Convert HIR to MLIR
    hir_to_mlir(codegen, &hir);
    if options.emit_mlir {
        println!("mlir:\n{}", codegen.module().as_operation());
    }

    // pass
    codegen.run_pass();

    // to llvm
    let m = codegen.to_inkwell_module(&inkwell_context);
    if options.emit_llvm_ir {
        println!("llvm:\n{}", m.to_string());
    }

    // run
    inkwell_codegen.add_module(m);
    inkwell_codegen.run_jit_eval_function_mlir(&module_name);
}
