use std::error::Error;

use crate::backend::llvm::CodeGen;


fn if_else(codegen: &CodeGen) -> Result<(), Box<dyn Error>> {
    let i32_type = codegen.context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let function = codegen.module.add_function("main", fn_type, None);
    let basic_block = codegen.context.append_basic_block(function, "entry");

    codegen.builder.position_at_end(basic_block);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    codegen.module.add_function("putchar", putchar_type, None);

    // print "Hi"
    let fun = codegen.module.get_function("putchar");
    codegen.builder.build_call(fun.unwrap(), &[i32_type.const_int(72, false).into()], "putchar");
    codegen.builder.build_call(fun.unwrap(), &[i32_type.const_int(105, false).into()], "putchar");

    // return value
    let value = &i32_type.const_int(42, false);
    codegen.builder.build_return(Some(value)).unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, execution_engine::JitFunction};

    use crate::backend::llvm::{codegen, link, target};

    use super::*;

    #[test]
    fn test() {
        let context = Context::create();
        let codegen = codegen::new(&context).unwrap();

        if_else(&codegen).unwrap();

        // show ll
        println!("----- Generated LLVM IR -----");
        println!("{}", codegen.module.to_string());
        println!("----- End of LLVM IR -----");

        // JIT
        type MainFunc = unsafe extern "C" fn() -> i32;
        unsafe {
            let main: JitFunction<MainFunc> = codegen.execution_engine.get_function("main").unwrap();
            let result = main.call();
            println!("result = {:?}", result);
        };
    }
}
