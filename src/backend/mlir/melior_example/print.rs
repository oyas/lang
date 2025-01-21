use std::mem;

use melior::{dialect::{arith, func, llvm::{self, attributes::Linkage}, memref, ods, DialectRegistry}, ir::{attribute::{DenseElementsAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute}, operation::OperationBuilder, r#type::{FunctionType, IntegerType, MemRefType, RankedTensorType}, Attribute, Block, BlockLike, Identifier, Location, Module, Operation, Region, Type, Value}, pass, utility::register_all_dialects, Context};
use mlir_sys::MlirOperation;

use crate::backend::mlir::CodeGen;


pub fn create_main(codegen: &CodeGen) {
    let context = &codegen.context;

    let location = Location::unknown(&context);
    let mut module = codegen.module.write().unwrap();

    let index_type = Type::index(&context);
    let i8_type = IntegerType::new(&context, 8).into();
    let i64_type = IntegerType::new(context, 64).into();
    let ptr_type = llvm::r#type::pointer(context, 0);

    // memref.global "private" constant @string : memref<14xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x57,0x6f,0x72,0x6c,0x64,0x21,0]>
    let mem_ref_type = MemRefType::new(i8_type, &[3], None, None);
    let memref = memref::global(
        context,
        "hello_string",
        Some("private"),
        mem_ref_type,
        Some(
            DenseElementsAttribute::new(
                RankedTensorType::new(&[3], i8_type, None).into(),
                &[
                    IntegerAttribute::new(i8_type, 72).into(),   // H
                    IntegerAttribute::new(i8_type, 105).into(),  // i
                    IntegerAttribute::new(i8_type, 0).into(),    // \0
                ],
            )
            .unwrap()
            .into(),
        ),
        true,
        None,
        location,
    );
    module.body().append_operation(memref);

    // llvm.func external @puts(!llvm.ptr) -> ()
    let puts_fn = llvm::func(
        context,
        StringAttribute::new(&context, "puts"),
        TypeAttribute::new(llvm::r#type::function(llvm::r#type::void(context), &[ptr_type], false)),
        Region::new(),
        &[(
            Identifier::new(&context, "linkage"),
            llvm::attributes::linkage(&context, Linkage::External),
        )],
        location,
    );
    let puts_fn = module.body().append_operation(puts_fn);

    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "main"),
        TypeAttribute::new(
            FunctionType::new(&context, &[], &[index_type]).into(),
        ),
        {
            let block = Block::new(&[]);

            let c = arith::constant(
                context,
                Attribute::parse(context, "41 : index").unwrap(),
                location,
            );
            let value = block.append_operation(c).result(0).unwrap().into();


            // %0 = memref.get_global @my_string : memref<14xi8>
            let v0 = block.append_operation(memref::get_global(context, "hello_string", mem_ref_type, location));
            // %1 = memref.extract_aligned_pointer_as_index %0 : memref<14xi8> -> index
            let v1 = block.append_operation(
                ods::memref::extract_aligned_pointer_as_index(context, index_type, v0.result(0).unwrap().into(), location).as_operation().clone()
            );
            // %2 = arith.index_cast %1 : index to i64
            let v2 = block.append_operation(
                arith::index_cast(v1.result(0).unwrap().into(), i64_type, location)
            );
            // %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
            let v3 = block.append_operation(
                ods::llvm::inttoptr(context, ptr_type, v2.result(0).unwrap().into(), location).as_operation().clone()
            );

            let f = block.append_operation(ods::llvm::mlir_addressof(
                context,
                ptr_type,
                FlatSymbolRefAttribute::new(&context, "puts"),
                location,
            ).as_operation().clone());
            let res = block.append_operation(ods::llvm::call(
                context,
                &[f.result(0).unwrap().into(), v3.result(0).unwrap().into()],
                location,
            ).as_operation().clone());

            // let res = block.append_operation(OperationBuilder::new("llvm.call", location)
            //     .add_results(&[])
            //     .add_attributes(&[(
            //         Identifier::new(&context, "callee"),
            //         FlatSymbolRefAttribute::new(&context, "puts").into(),
            //     )])
            //     .add_operands(&[v3.result(0).unwrap().into()])
            //     .build()
            //     .unwrap()
            // );

            block.append_operation(func::r#return(&[value], location));

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

    assert!(module.as_operation().verify());
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use inkwell::llvm_sys;
    use mlir_sys::{mlirTranslateModuleToLLVMIR, LLVMOpaqueContext};

    use super::*;

    #[test]
    fn test() {
        let context = Context::new();
        let mut codegen = CodeGen::new(&context);
        create_main(&codegen);
        println!("mlir (before pass): {}", codegen.module.read().unwrap().as_operation());
        codegen.run_pass();
        println!("mlir (after pass): {}", codegen.module.read().unwrap().as_operation());

        // print llvm ir
        codegen.save_llvm_ir("", true);

        // execute
        let result = codegen.execute_no_args("main");
        println!("result = {:?}", result);

        // to inkwell module
        let inkwell_context = inkwell::context::Context::create();
        let inkwell_module = codegen.to_inkwell_module(&inkwell_context);
        println!("inkwell_module = {}", inkwell_module.to_string());

        // save object file
        // codegen.save_object("print.o");
    }
}
