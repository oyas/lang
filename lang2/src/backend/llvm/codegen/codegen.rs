use std::error::Error;

use inkwell::{builder::Builder, context::Context, execution_engine::ExecutionEngine, module::Module, OptimizationLevel};

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
}

pub fn new(context: &Context) -> Result<CodeGen<'_>, Box<dyn Error>> {
    // let context = Context::create();
    let module = context.create_module("main");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let builder = context.create_builder();
    let codegen = CodeGen {
        context,
        module,
        builder,
        execution_engine,
    };

    Ok(codegen)
}

impl<'ctx> CodeGen<'ctx> {
}