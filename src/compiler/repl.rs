use std::io::Write;

use inkwell::{context::Context, values::{BasicValue, BasicValueEnum, IntValue, PointerValue}, AddressSpace};

use crate::{ast::{Ast, IndentedStatement, Statement}, backend::llvm::{codegen, CodeGen}, parser::{self, is_complete}};

use super::Options;


pub fn repl(options: &Options) {
    let context = Context::create();
    let mut codegen = setup(&context);
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
        let result = eval(&mut codegen, &ast, options);
        println!("{}", result);
    }
}

const GLOBAL_OUTPUT_PTR_NAME: &str = "__global_output_ptr";

pub fn eval<'a>(codegen: &mut CodeGen<'a>, ast: &Ast, options: &Options) -> String {
    // setup eval function
    let num = codegen.modules.len();
    let module_name = format!("eval_{}", num);
    let module = codegen.create_module(&module_name, &module_name);
    let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let fn_type = ptr_type.fn_type(&[], false);
    let fn_name = format!("eval_{}", num);
    let eval_function = module.add_function(&fn_name, fn_type, None);
    let entry_block = codegen.context.append_basic_block(eval_function, "entry");
    codegen.builder.position_at_end(entry_block);

    // generate llvm ir
    let mut last_int_value = None;
    for IndentedStatement(_i, s) in &ast.0 {
        last_int_value = match s {
            Statement::Let(e) => Some(codegen::calc::build_expression(&codegen, e).unwrap()),
            Statement::Assign(e) => Some(codegen::calc::build_expression(&codegen, e).unwrap()),
            Statement::Expr(e) => Some(codegen::calc::build_expression(&codegen, e).unwrap()),
            Statement::Function(_) => panic!("Not implement! (eval)"),
            //_ => panic!("Not implement! (eval)"),
        };
    };

    let global_output_ptr = print_to_output(codegen, last_int_value.unwrap());

    codegen.builder.build_return(Some(&global_output_ptr)).unwrap();

    if options.emit_llvm_ir {
        println!("----- Generated LLVM IR -----");
        println!("{}", module.to_string());
        println!("----- End of LLVM IR -----");
    }

    // execute
    codegen.run_jit_eval_function(&fn_name)
}

pub fn setup(context: &Context) -> CodeGen {
    let codegen = CodeGen::new(&context);

    codegen
}

fn make_output_buffer<'a>(codegen: &CodeGen<'a>) -> PointerValue<'a> {
    let module = &codegen.current_module;
    let i32_type = codegen.context.i32_type();
    let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let i8_type = codegen.context.i8_type();
    let i8_arr_type = i8_type.array_type(4096);
    let global_str = module.add_global(i8_arr_type, None, "global_output_str");
    global_str.set_initializer(&i8_arr_type.const_zero());

    let zero = i32_type.const_int(0, false);
    unsafe {
        codegen.builder.build_gep(ptr_type, global_str.as_pointer_value(), &[zero, zero], "global_output_ptr").unwrap()
    }
}

fn print_to_output<'a>(codegen: &CodeGen<'a>, value: IntValue) -> PointerValue<'a> {
    let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let i32_type = codegen.context.i32_type();
    let sprintf_fn_type = i32_type.fn_type(&[ptr_type.into(), ptr_type.into(), i32_type.into()], true);
    let sprintf_fn = codegen.current_module.add_function("sprintf", sprintf_fn_type, None);

    if !codegen.values.borrow().contains_key(GLOBAL_OUTPUT_PTR_NAME) {
        let buffer = make_output_buffer(codegen);
        let mut values = codegen.values.borrow_mut();
        values.insert(GLOBAL_OUTPUT_PTR_NAME.to_string(), buffer.as_basic_value_enum());
    }
    let values = codegen.values.borrow();
    let global_output_ptr_op = values.get(GLOBAL_OUTPUT_PTR_NAME);
    let BasicValueEnum::PointerValue(ref global_output_ptr) = global_output_ptr_op.unwrap() else {
        panic!("Wrong type of __global_output_ptr.")
    };
    let fmt_str = codegen.builder.build_global_string_ptr("%d", "fmt_str").unwrap();
    codegen.builder.build_call(
        sprintf_fn,
        &[(*global_output_ptr).into(), fmt_str.as_pointer_value().into(), value.into()],
        "sprintf_call"
    ).unwrap();

    *global_output_ptr
}
