use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, IntValue};
use inkwell::OptimizationLevel;

use std::error::Error;
use std::path::Path;

use super::{link, target};

type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        // let i64_type = self.context.i64_type();
        let i32_type = self.context.i32_type();
        // let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let fn_type = i32_type.fn_type(&[], false);
        let function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        // let x = function.get_nth_param(0)?.into_int_value();
        // let y = function.get_nth_param(1)?.into_int_value();
        // let z = function.get_nth_param(2)?.into_int_value();

        // let sum = self.builder.build_int_add(x, y, "sum").unwrap();
        // let sum = self.builder.build_int_add(sum, z, "sum").unwrap();

        // declare i32 @putchar(i32)
        let putchar_type = i32_type.fn_type(&[i32_type.into()], false);
        self.module.add_function("putchar", putchar_type, None);

        // call i32 @putchar(i32 72)
        let fun = self.module.get_function("putchar");
        self.builder.build_call(fun.unwrap(), &[i32_type.const_int(72, false).into()], "putchar");
        self.builder.build_call(fun.unwrap(), &[i32_type.const_int(105, false).into()], "putchar");

        // let ret = Some(&sum);
        // let value = i64_type.create_generic_value(42, false);
        let value = &i32_type.const_int(42, false);
        let aa = Some(value as &dyn BasicValue);
        self.builder.build_return(aa).unwrap();
        // self.builder.build_return(Some(&i32_type.const_int(42, false))).unwrap();

        println!("{}", self.module.to_string());
        let path = Path::new("module.bc");
        self.module.write_bitcode_to_path(path);

        // unsafe { self.execution_engine.get_function("sum").ok() }
        None
    }
}

pub fn compile() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    // let sum = codegen.jit_compile_sum().ok_or("Unable to JIT compile `sum`")?;
    let sum = codegen.jit_compile_sum();

    target::run_passes_on(&codegen.module, target::Triple::Default);
    let objects = target::write_to_file(&codegen.module, target::Triple::Default, "main");
    link::link(&objects).unwrap();

    let x = 1u64;
    let y = 2u64;
    let z = 3u64;

    // unsafe {
    //     println!("{} + {} + {} = {}", x, y, z, sum.call(x, y, z));
    //     assert_eq!(sum.call(x, y, z), x + y + z);
    // }

    Ok(())
}
