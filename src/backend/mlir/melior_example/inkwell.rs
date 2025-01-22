use std::ffi::CStr;

use attribute::FlatSymbolRefAttribute;
use inkwell::llvm_sys;
use melior::{
    dialect::{arith, func, index, DialectRegistry},
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

use crate::backend::mlir::CodeGen;

pub fn create_add(codegen: &mut CodeGen) {
    let context = codegen.context;
    let location = Location::unknown(&context);
    let mut arc_module = codegen.new_module("add");
    let module = arc_module.write().unwrap();

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
        &[],
        location,
    ));

    println!("{}", module.as_operation());
    assert!(module.as_operation().verify());
}

pub fn create_main(codegen: &mut CodeGen) {
    let context = codegen.context;
    let location = Location::unknown(&context);
    let mut arc_module = codegen.new_module("main");
    let module = arc_module.write().unwrap();

    let index_type = Type::index(&context);
    let i32_type = IntegerType::new(&context, 32).into();

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
            FunctionType::new(&context, &[], &[i32_type]).into(),
        ),
        {
            let block = Block::new(&[]);

            let a = Attribute::parse(context, "40 : index");
            let c = arith::constant(
                context,
                a.unwrap(),
                location,
            );
            let value = block.append_operation(c).result(0).unwrap().into();

            let sum = block.append_operation(func::call(
                context,
                FlatSymbolRefAttribute::new(&context, "add"),
                &[value, value],
                &[index_type],
                location,
            ));
            let s = block.append_operation(index::casts(sum.result(0).unwrap().into(), i32_type, location));

            block.append_operation(func::r#return(&[s.result(0).unwrap().into()], location));

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
}

#[cfg(test)]
mod tests {
    use crate::backend::{self, llvm::{self, target}};

    use super::*;

    #[test]
    fn test() {
        let context = Context::new();
        let inkwell_context = inkwell::context::Context::create();
        let mut codegen = CodeGen::new(&context);
        let mut inkwell_codegen = backend::llvm::CodeGen::new(&inkwell_context);

        // add module
        create_add(&mut codegen);
        codegen.run_pass();
        let module_add = codegen.to_inkwell_module(&inkwell_context);
        inkwell_codegen.add_module(module_add);

        // main module
        create_main(&mut codegen);
        codegen.run_pass();
        let module_main = codegen.to_inkwell_module(&inkwell_context);
        inkwell_codegen.add_module(module_main);

        // JIT
        let result = inkwell_codegen.run_jit();
        println!("result = {:?}", result);

        // compile
        inkwell_codegen.compile().unwrap();  // for default target
        inkwell_codegen.compile_with_target(&target::Triple::Wasm32WasiLibc).unwrap();  // for WASM
    }
}
