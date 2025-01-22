use std::ffi::CStr;

use attribute::FlatSymbolRefAttribute;
use inkwell::llvm_sys;
use melior::{
    dialect::{arith, func, DialectRegistry},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        *,
    },
    pass,
    utility::register_all_dialects,
    Context, ExecutionEngine,
};
use mlir_sys::{mlirTranslateModuleToLLVMIR, LLVMOpaqueContext};
use r#type::IntegerType;

pub fn create_add(context: &Context) -> Module {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let index_type = Type::index(&context);

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "add"),
        TypeAttribute::new(
            FunctionType::new(&context, &[index_type, index_type], &[index_type]).into(),
        ),
        {
            let block = Block::new(&[(index_type, location), (index_type, location)]);

            let sum = block.append_operation(arith::addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[
            (
                Identifier::new(&context, "llvm.emit_c_interface"),
                Attribute::unit(&context),
            ), // required to execute invoke_packed
        ],
        location,
    ));

    assert!(module.as_operation().verify());
    println!("{}", module.as_operation());

    // pass
    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert_eq!(pass_manager.run(&mut module), Ok(()));
    println!("{}", module.as_operation());

    module
}

pub fn create_main(context: &Context) -> Module {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    // let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let index_type = Type::index(&context);
    // let integer_type = IntegerType::new(&context, 64).into();

    let add_fn = func::func(
        context,
        StringAttribute::new(&context, "add"),
        TypeAttribute::new(
            FunctionType::new(&context, &[index_type, index_type], &[index_type]).into(),
        ),
        Region::new(),
        &[
            (
                Identifier::new(&context, "sym_visibility"),
                Attribute::parse(&context, r#""private""#).unwrap(),
            ),
        ],
        location,
    );
    module.body().append_operation(add_fn);
    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "main"),
        TypeAttribute::new(
            FunctionType::new(&context, &[], &[index_type]).into(),
        ),
        {
            let block = Block::new(&[]);

            let a = Attribute::parse(context, "41 : index");
            println!("a = {:?}", a);
            let c = arith::constant(
                context,
                a.unwrap(),
                location,
            );
            let value = block.append_operation(c).result(0).unwrap().into();
            // let value = c_42.result(0).unwrap().into();

            let sum = block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(&context, "add"),
                &[value, value],
                &[index_type],
                location,
            ));

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[
            (
                Identifier::new(&context, "llvm.emit_c_interface"),
                Attribute::unit(&context),
            ),  // required to execute invoke_packed
        ],
        location,
    ));

    println!("{}", module.as_operation());
    assert!(module.as_operation().verify());

    // pass
    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert_eq!(pass_manager.run(&mut module), Ok(()));
    println!("{}", module.as_operation());

    module
}

fn execute(context: &Context, module: &Module, module_main: &Module) {
    // execute
    let engine = ExecutionEngine::new(&module, 0, &[], true);
    let mut argument1: usize = 42;
    let mut argument2: usize = 43;
    let mut result: usize = 0;
    let res = unsafe {
        engine.invoke_packed(
            "add",
            &mut [
                &mut argument1 as *mut usize as *mut (),
                &mut argument2 as *mut usize as *mut (),
                &mut result as *mut usize as *mut (),
            ],
        )
    };
    assert_eq!(res, Ok(()));
    println!("result = {}", result);

    // emit llvm ir (inkwell)
    // let llvm_context = inkwell::context::Context::create();
    // let llvm_module = unsafe {
    //     let m = mlirTranslateModuleToLLVMIR(module.as_operation().to_raw(), llvm_context.raw() as *mut LLVMOpaqueContext);
    //     inkwell::module::Module::new(m as *mut inkwell::llvm_sys::LLVMModule)
    // };
    // println!("----- Generated LLVM IR -----");
    // println!("{}", llvm_module.to_string());
    // println!("----- End of LLVM IR -----");

    // emit llvm ir (llvm-sys)
    let llvm_module = unsafe {
        let llvm_context = llvm_sys::core::LLVMContextCreate();
        let llvm_module = mlirTranslateModuleToLLVMIR(
            module.as_operation().to_raw(),
            llvm_context as *mut LLVMOpaqueContext,
        );
        let ir_str = llvm_sys::core::LLVMPrintModuleToString(llvm_module as *mut llvm_sys::LLVMModule);
        CStr::from_ptr(ir_str).to_str().unwrap()
    };
    println!("----- Generated LLVM IR -----");
    println!("{}", llvm_module);
    println!("----- End of LLVM IR -----");

    // for link
    let symbol = engine.lookup("add");

    // main module emit llvm ir (llvm-sys)
    let llvm_module = unsafe {
        let llvm_context = llvm_sys::core::LLVMContextCreate();
        let llvm_module = mlirTranslateModuleToLLVMIR(
            module_main.as_operation().to_raw(),
            llvm_context as *mut LLVMOpaqueContext,
        );
        let ir_str = llvm_sys::core::LLVMPrintModuleToString(llvm_module as *mut llvm_sys::LLVMModule);
        CStr::from_ptr(ir_str).to_str().unwrap()
    };
    println!("----- Generated LLVM IR (main) -----");
    println!("{}", llvm_module);
    println!("----- End of LLVM IR -----");

    // main module
    let engine = ExecutionEngine::new(&module_main, 0, &[], true);
    let mut result: usize = 0;
    let res = unsafe {
        engine.register_symbol("add", symbol);
        engine.invoke_packed(
            "main",
            &mut [
                &mut result as *mut usize as *mut (),
            ],
        )
    };
    assert_eq!(res, Ok(()));
    println!("result = {}", result);

    // emit obj
    engine.dump_to_object_file("test.o");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let context = Context::new();
        let module = create_add(&context);
        let module_main = create_main(&context);
        execute(&context, &module, &module_main);
    }
}
