use std::error::Error;

use inkwell::{values::BasicValueEnum, IntPredicate};

use crate::backend::llvm::CodeGen;


fn if_else(codegen: &CodeGen) -> Result<(), Box<dyn Error>> {
    let module = codegen.get_main_module();
    let i32_type = codegen.context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let basic_block = codegen.context.append_basic_block(main_function, "entry");

    codegen.builder.position_at_end(basic_block);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    let fun = module.add_function("putchar", putchar_type, None);

    // print "Hi"
    // let fun = codegen.module.get_function("putchar").unwrap();
    codegen.builder.build_call(fun, &[i32_type.const_int(72, false).into()], "putchar");
    let call_site_value = codegen.builder.build_call(fun, &[i32_type.const_int(105, false).into()], "putchar").unwrap();
    let return_value = call_site_value.try_as_basic_value().left().unwrap();
    let BasicValueEnum::IntValue(int_value) = return_value else {
        panic!("Returned value can't unwrap: {:?}", return_value);
    };
    println!("return_value: {:?}", return_value);
    // codegen.builder.build_call(fun, &[return_value.into()], "putchar");

    // blocks
    let then_block = codegen.context.append_basic_block(main_function, "then");
    let else_block = codegen.context.append_basic_block(main_function, "else");
    let end_block = codegen.context.append_basic_block(main_function, "end");

    // br
    let comp_return_value_eq_0 = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        int_value,
        i32_type.const_int(0, false),
        "comp_return_value_eq_0",
    ).unwrap();
    codegen.builder.build_conditional_branch(comp_return_value_eq_0, then_block, else_block);

    // then
    codegen.builder.position_at_end(then_block);
    let a_then = i32_type.const_int(10, false);
    codegen.builder.build_unconditional_branch(end_block);

    // else
    codegen.builder.position_at_end(else_block);
    let a_else = i32_type.const_int(20, false);
    codegen.builder.build_unconditional_branch(end_block);

    // end
    codegen.builder.position_at_end(end_block);
    let phi = codegen.builder.build_phi(i32_type, "a").unwrap();
    phi.add_incoming(&[
        (&a_then, then_block),
        (&a_else, else_block),
    ]);
    let BasicValueEnum::IntValue(a) = phi.as_basic_value() else {
        panic!("a is not IntValue. {:?}", phi);
    };
    codegen.builder.build_return(Some(&a)).unwrap();

    Ok(())
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, execution_engine::JitFunction};

    use super::*;

    #[test]
    fn test() {
        let context = Context::create();
        let codegen = CodeGen::new(&context);

        if_else(&codegen).unwrap();

        // show ll
        println!("----- Generated LLVM IR -----");
        println!("{}", codegen.get_main_module().to_string());
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
