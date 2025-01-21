use std::error::Error;

use inkwell::{context::Context, OptimizationLevel};

use crate::backend::llvm::CodeGen;


fn hello_world(codegen: &CodeGen) -> Result<(), Box<dyn Error>> {
    let module = codegen.get_main_module();
    let i32_type = codegen.context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let function = module.add_function("main", fn_type, None);
    let basic_block = codegen.context.append_basic_block(function, "entry");

    codegen.builder.position_at_end(basic_block);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    let fun = module.add_function("putchar", putchar_type, None);

    // print "Hi"
    codegen.builder.build_call(fun, &[i32_type.const_int(72, false).into()], "putchar");
    codegen.builder.build_call(fun, &[i32_type.const_int(105, false).into()], "putchar");

    // return value
    let value = &i32_type.const_int(42, false);
    codegen.builder.build_return(Some(value)).unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use inkwell::execution_engine::JitFunction;

    use crate::backend::llvm::target;

    use super::*;

    #[test]
    fn test() {
        let context = Context::create();
        let codegen = CodeGen::new(&context);

        hello_world(&codegen).unwrap();

        // JIT
        type MainFunc = unsafe extern "C" fn() -> i32;
        unsafe {
            let main: JitFunction<MainFunc> = codegen.execution_engine.get_function("main").unwrap();
            let result = main.call();
            println!("result = {:?}", result);
        };

        // default target
        codegen.compile().unwrap();

        // wasm32 target
        codegen.compile_with_target(&target::Triple::Wasm32WasiLibc).unwrap();
    }
}
