use std::error::Error;

use inkwell::{values::{AsValueRef, BasicValueEnum}, AddressSpace, IntPredicate};

use crate::backend::llvm::CodeGen;


fn getelementptr(codegen: &CodeGen) -> Result<(), Box<dyn Error>> {
    let module = codegen.get_main_module();
    let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let i32_type = codegen.context.i32_type();
    let fn_type = i32_type.fn_type(&[], false);
    let main_function = module.add_function("main", fn_type, None);
    let entry_block = codegen.context.append_basic_block(main_function, "entry");
    let zero = i32_type.const_int(0, false);
    let two = i32_type.const_int(2, false);

    // declare i32 @putchar(i32)
    let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
    let fun = module.add_function("putchar", putchar_type, None);

    // blocks
    let alloca_block = codegen.context.append_basic_block(main_function, "alloca");
    let alloca2_block = codegen.context.append_basic_block(main_function, "alloca2");
    let copy_block = codegen.context.append_basic_block(main_function, "copy");
    let malloc_block = codegen.context.append_basic_block(main_function, "malloc");
    let array_block = codegen.context.append_basic_block(main_function, "array");
    let struct_block = codegen.context.append_basic_block(main_function, "struct");

    // print "Hi"
    codegen.builder.position_at_end(entry_block);
    codegen.builder.build_call(fun, &[i32_type.const_int(72, false).into()], "putchar");
    let call_site_value = codegen.builder.build_call(fun, &[i32_type.const_int(105, false).into()], "putchar").unwrap();
    let return_value = call_site_value.try_as_basic_value().left().unwrap();
    let BasicValueEnum::IntValue(int_value) = return_value else {
        panic!("Returned value can't unwrap: {:?}", return_value);
    };
    codegen.builder.build_unconditional_branch(struct_block);

    // alloca
    codegen.builder.position_at_end(alloca_block);
    let ptr = codegen.builder.build_alloca(i32_type, "ptr").unwrap();
    codegen.builder.build_store(ptr, i32_type.const_int(40, false));
    let value = codegen.builder.build_load(i32_type, ptr, "value").unwrap().into_int_value();
    codegen.builder.build_return(Some(&value)).unwrap();

    // alloca2
    codegen.builder.position_at_end(alloca2_block);
    let ptr = codegen.builder.build_alloca(i32_type, "ptr").unwrap();
    let ptr_ptr = codegen.builder.build_alloca(ptr_type, "ptr_ptr").unwrap();
    codegen.builder.build_store(ptr, i32_type.const_int(41, false));
    codegen.builder.build_store(ptr_ptr, ptr);
    let loaded_ptr = codegen.builder.build_load(ptr_type, ptr_ptr, "loaded_ptr").unwrap().into_pointer_value();
    let value = codegen.builder.build_load(i32_type, loaded_ptr, "value").unwrap().into_int_value();
    codegen.builder.build_return(Some(&value)).unwrap();

    // copy
    codegen.builder.position_at_end(copy_block);
    let var1 = codegen.builder.build_alloca(i32_type, "var1").unwrap();
    let var2 = codegen.builder.build_alloca(i32_type, "var2").unwrap();
    let value_43 = i32_type.const_int(43, false);
    codegen.builder.build_store(var1, value_43);
    let loaded_value = codegen.builder.build_load(i32_type, var1, "loaded_value").unwrap().into_int_value();
    codegen.builder.build_store(var2, loaded_value);
    let value = codegen.builder.build_load(i32_type, var2, "loaded_value").unwrap().into_int_value();
    codegen.builder.build_return(Some(&value)).unwrap();

    // malloc
    codegen.builder.position_at_end(malloc_block);
    let var = codegen.builder.build_malloc(i32_type, "var").unwrap();
    codegen.builder.build_store(var, i32_type.const_int(50, false));
    let value = codegen.builder.build_load(i32_type, var, "value").unwrap().into_int_value();
    codegen.builder.build_return(Some(&value)).unwrap();

    // array
    codegen.builder.position_at_end(array_block);
    let i32_array5_type = i32_type.array_type(5);
    let array = codegen.builder.build_alloca(i32_array5_type, "array").unwrap();
    // let array = codegen.builder.build_array_alloca(i32_type, two, "array").unwrap();
    let elem_ptr = unsafe {
        codegen.builder.build_gep(i32_array5_type, array, &[zero, two], "elem_ptr").unwrap()
        // codegen.builder.build_gep(i32_type, array, &[two], "elem_ptr").unwrap()
    };
    codegen.builder.build_store(elem_ptr, i32_type.const_int(42, false));
    let value = codegen.builder.build_load(i32_type, elem_ptr, "value").unwrap();
    codegen.builder.build_return(Some(&value)).unwrap();

    // struct
    codegen.builder.position_at_end(struct_block);
    let field_types = &[i32_type.into(), i32_type.into()];
    let struct_ty = codegen.context.struct_type(field_types, false);
    let s = codegen.builder.build_alloca(struct_ty, "s").unwrap();
    let elem_ptr = codegen.builder.build_struct_gep(struct_ty, s, 1, "elem_ptr").unwrap();
    codegen.builder.build_store(elem_ptr, i32_type.const_int(60, false));
    let value = codegen.builder.build_load(i32_type, elem_ptr, "value").unwrap();
    codegen.builder.build_return(Some(&value)).unwrap();

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

        getelementptr(&codegen).unwrap();

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
