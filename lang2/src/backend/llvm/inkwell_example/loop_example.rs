use std::error::Error;

use inkwell::{values::BasicValueEnum, IntPredicate};

use crate::backend::llvm::CodeGen;


fn loop_example(codegen: &CodeGen) -> Result<(), Box<dyn Error>> {
    let module = codegen.get_main_module();
    let i32_type = codegen.context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = codegen.context.append_basic_block(main_function, "entry");
    let zero = i32_type.const_int(0, false);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    let fun = module.add_function("putchar", putchar_type, None);

    // blocks
    let loop_block = codegen.context.append_basic_block(main_function, "loop");
    let loop_body_block = codegen.context.append_basic_block(main_function, "loop_body");
    let end_block = codegen.context.append_basic_block(main_function, "end");

    // print "Hi"
    codegen.builder.position_at_end(entry_block);
    codegen.builder.build_call(fun, &[i32_type.const_int(72, false).into()], "putchar");
    let call_site_value = codegen.builder.build_call(fun, &[i32_type.const_int(105, false).into()], "putchar").unwrap();
    let return_value = call_site_value.try_as_basic_value().left().unwrap();
    let BasicValueEnum::IntValue(int_value) = return_value else {
        panic!("Returned value can't unwrap: {:?}", return_value);
    };
    codegen.builder.build_unconditional_branch(loop_block);

    // loop
    codegen.builder.position_at_end(loop_block);
    let phi = codegen.builder.build_phi(i32_type, "i").unwrap();
    phi.add_incoming(&[
        (&zero, entry_block),
    ]);
    let BasicValueEnum::IntValue(i) = phi.as_basic_value() else {
        panic!("i is not IntValue. {:?}", phi);
    };
    let comp_i_eq_10 = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        i,
        i32_type.const_int(10, false),
        "comp_i_eq_10",
    ).unwrap();
    codegen.builder.build_conditional_branch(comp_i_eq_10, end_block, loop_body_block);

    // loop_body
    codegen.builder.position_at_end(loop_body_block);
    codegen.builder.build_call(fun, &[i32_type.const_int(120, false).into()], "putchar");
    codegen.builder.build_call(fun, &[i32_type.const_int(10, false).into()], "putchar");
    let i_next = codegen.builder.build_int_add(i, i32_type.const_int(1, false), "i_next").unwrap();
    phi.add_incoming(&[(&i_next, loop_body_block)]);
    codegen.builder.build_unconditional_branch(loop_block);

    // end
    codegen.builder.position_at_end(end_block);
    codegen.builder.build_return(Some(&i)).unwrap();

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

        loop_example(&codegen).unwrap();

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
