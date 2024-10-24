use std::error::Error;

use inkwell::{module::Module, values::BasicValueEnum, IntPredicate};

use crate::backend::llvm::CodeGen;


fn module_example<'a>(codegen: &mut CodeGen<'a>) -> Result<(), Box<dyn Error>> {
    let i32_type = codegen.context.i32_type();
    let zero = i32_type.const_int(0, false);
    let module = codegen.get_main_module();
    let module2 = codegen.create_module("module2", "module2");

    // main function
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);

    // f1 function
    let f1_type = i32_type.fn_type(&[i32_type.into()], false);
    let f1 = module2.add_function("f1", f1_type, None);
    let f1_main = module.add_function("f1", f1_type, None);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    let fun = module.add_function("putchar", putchar_type, None);
    let fun2 = module2.add_function("putchar", putchar_type, None);

    // blocks
    let entry_block = codegen.context.append_basic_block(main_function, "entry");
    let f1_entry_block = codegen.context.append_basic_block(f1, "entry");

    // print "Hi"
    codegen.builder.position_at_end(entry_block);
    codegen.builder.build_call(fun2, &[i32_type.const_int(72, false).into()], "putchar").unwrap();
    codegen.builder.build_call(fun2, &[i32_type.const_int(105, false).into()], "putchar").unwrap();
    let call_site_value = codegen.builder.build_call(f1, &[i32_type.const_int(65, false).into()], "f1").unwrap();
    let return_value = call_site_value.try_as_basic_value().left().unwrap();
    let BasicValueEnum::IntValue(int_value) = return_value else {
        panic!("Returned value can't unwrap: {:?}", return_value);
    };
    codegen.builder.build_return(Some(&int_value)).unwrap();

    // f1 function
    codegen.builder.position_at_end(f1_entry_block);
    let param = f1.get_first_param().unwrap();
    codegen.builder.build_call(fun2, &[param.into()], "putchar");
    codegen.builder.build_return(Some(&param)).unwrap();

    // another f1 function
    // let module3 = codegen.create_module("module3", "module3");
    // let f1_3 = module3.add_function("f1", f1_type, None);
    // let module3_entry_block = codegen.context.append_basic_block(f1_3, "entry");
    // codegen.builder.position_at_end(module3_entry_block);
    // codegen.builder.build_return(Some(&zero)).unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, execution_engine::JitFunction, llvm_sys::target_machine, targets::FileType};

    use crate::backend::llvm::{codegen, link, target};

    use super::*;

    #[test]
    fn test() {
        let context = Context::create();
        let mut codegen = codegen::new(&context).unwrap();

        module_example(&mut codegen).unwrap();

        // JIT
        let result = codegen.run_jit();
        println!("result = {:?}", result);

        // compile
        codegen.compile().unwrap();  // for default target
        codegen.compile_with_target(&target::Triple::Wasm32WasiLibc).unwrap();  // for WASM
    }
}
