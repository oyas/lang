use std::{fs::File, io::{BufRead, BufReader}};

use inkwell::context::Context;

use crate::parser::{self};

use super::repl;

pub struct Options {
    pub emit_llvm_ir: bool,
    pub emit_mlir: bool,
    pub emit_hir: bool,
    pub emit_ast: bool,
}

pub fn compile(file_name: &str, options: &Options) {
    let mut buf_reader = match File::open(file_name) {
        Ok(file) => BufReader::new(file),
        Err(e) => panic!("Error occurred while reading {}: {}", file_name, e),
    };

    let context = Context::create();
    let mut codegen = repl::setup(&context);

    let mut buffer = String::new();
    loop {
        match buf_reader.read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let ast = match parser::parse(&buffer) {
                    Ok((_, r)) => r,
                    Err(e) => {
                        println!("parse error! {:?}", e);
                        continue;
                    }
                };
                if options.emit_ast {
                    println!("parsed: {:?}", ast);
                }
                let result = repl::eval(&mut codegen, &ast, options);
                println!("{}", result);
                buffer.clear();
            }
            Err(e) => {
                println!("Error occurred while read_tokens: {}", e);
                break;
            }
        }
    }

    // main function
    let module = codegen.get_main_module();
    // let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let i32_type = codegen.context.i32_type();
    let zero = i32_type.const_int(0, false);
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = codegen.context.append_basic_block(main_function, "entry");
    codegen.builder.position_at_end(entry_block);
    // let eval_fn_type = ptr_type.fn_type(&[], false);
    // let eval_fn = module.add_function("eval_1", eval_fn_type, None);
    // let eval_result = codegen.builder.build_call(eval_fn, &[], "eval_call").unwrap();
    // let return_value = eval_result.try_as_basic_value().left().unwrap();
    // let BasicValueEnum::PointerValue(result_ptr) = return_value else {
    //     panic!("Returned value can't unwrap: {:?}", return_value);
    // };
    // let fmt_str = codegen.builder.build_global_string_ptr("%s", "fmt_str").unwrap();
    // let printf_fn_type = i32_type.fn_type(&[ptr_type.into(), i32_type.into()], true);
    // let printf_fn = module.add_function("printf", printf_fn_type, None);
    // codegen.builder.build_call(
    //     printf_fn,
    //     &[fmt_str.as_pointer_value().into(), result_ptr.into()],
    //     "printf_call"
    // ).unwrap();
    codegen.builder.build_return(Some(&zero)).unwrap();

    if options.emit_llvm_ir {
        println!("----- Generated LLVM IR -----");
        println!("{}", module.to_string());
        println!("----- End of LLVM IR -----");
    }

    codegen.compile().unwrap();
}