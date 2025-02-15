use std::sync::{Arc, RwLock};

use melior::{dialect::{arith, func, llvm, memref, ods::{self}}, ir::{attribute::{DenseElementsAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute}, r#type::{FunctionType, IntegerType, MemRefType, RankedTensorType}, Attribute, Block, BlockLike, Identifier, Location, OperationRef, Region, Type}, Context};

use crate::{backend::mlir::CodeGen, hir::{self, BuiltinFunction, Expression, ExpressionBody, Hir}};

pub fn hir_to_mlir(codegen: &mut CodeGen, hir: &Hir) {
    let module_arc = codegen.new_module(&hir.module.read().unwrap().finename);
    let module = module_arc.write().unwrap();
    let context = &codegen.context;
    let block = module.body();
    let binding = hir.read();
    let binding = binding.pre_body.read().unwrap();
    let exprs_pre_body = binding.blocks.get(0).unwrap().exprs.iter();
    let binding = hir.read();
    let binding = binding.body.read().unwrap();
    let exprs_body = binding.blocks.get(0).unwrap().exprs.iter();
    let exprs = exprs_pre_body.chain(exprs_body);
    for expr in exprs {
        match &expr.read().unwrap().body {
            ExpressionBody::Function{..} => function_to_mlir(&context, &block, &expr.read().unwrap()),
            ExpressionBody::BuiltinFunction(..) => builtin_function_to_mlir(&context, &block, &expr.read().unwrap()),
            ExpressionBody::Str{..} => str_to_mlir(&context, &block, &expr.read().unwrap()),
            _ => panic!("Not implemented {:?}", expr),
        };
    }
}

fn region_to_mlir<'a>(context: &'a Context, region: &Arc<RwLock<hir::Region>>) -> Region<'a> {
    let ret = Region::new();
    for block in &region.read().unwrap().blocks {
        ret.append_block(block_to_mlir(context, block));
    }
    ret
}

fn block_to_mlir<'a>(context: &'a Context, block: &hir::Block) -> Block<'a> {
    let ret = Block::new(&[]);
    let mut last_result = None;
    for expr in &block.exprs {
        last_result = Some(expr_to_mlir(context, &ret, &expr.read().unwrap()));
    }
    match block.terminator {
        Some(hir::Terminator::Return) => {
            let operation = func::r#return(&[last_result.unwrap().result(0).unwrap().into()], location_to_mlir(&context, &block.location));
            ret.append_operation(operation);
        },
        None => {}
        _ => panic!("Not implemented"),
    }
    ret
}

fn expr_to_mlir<'a>(context: &'a Context, block: &'a Block<'a>, expr: &Expression) -> OperationRef<'a, 'a> {
    let location = location_to_mlir(context, &expr.location);
    let operation = match &expr.body {
        ExpressionBody::I64(i) => {
            let value = Attribute::parse(context, &format!("{} : i64", i)).unwrap();
            arith::constant(context, value, location_to_mlir(context, &expr.location))
        },
        ExpressionBody::Str { name, text } => {
            let size = text.len() + 1;
            let i8_type = IntegerType::new(&context, 8).into();
            let i64_type = IntegerType::new(context, 64).into();
            let index_type = Type::index(&context);
            let ptr_type = llvm::r#type::pointer(context, 0);
            let mem_ref_type = MemRefType::new(i8_type, &[size as i64], None, None);
            // %0 = memref.get_global @name : memref<sizexi8>
            let v0 = block.append_operation(memref::get_global(context, name, mem_ref_type, location));
            // %1 = memref.extract_aligned_pointer_as_index %0 : memref<sizexi8> -> index
            let v1 = block.append_operation(
                ods::memref::extract_aligned_pointer_as_index(context, index_type, v0.result(0).unwrap().into(), location).as_operation().clone()
            );
            // %2 = arith.index_cast %1 : index to i64
            let v2 = block.append_operation(
                arith::index_cast(v1.result(0).unwrap().into(), i64_type, location)
            );
            // %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
            ods::llvm::inttoptr(context, ptr_type, v2.result(0).unwrap().into(), location).as_operation().clone()
        }
        ExpressionBody::Add(a, b) => {
            let a = expr_to_mlir(context, block, &a.read().unwrap());
            let b = expr_to_mlir(context, block, &b.read().unwrap());
            arith::addi(
                a.result(0).unwrap().into(),
                b.result(0).unwrap().into(),
                location_to_mlir(context, &expr.location),
            )
        },
        ExpressionBody::Sub(a, b) => {
            let a = expr_to_mlir(context, block, &a.read().unwrap());
            let b = expr_to_mlir(context, block, &b.read().unwrap());
            arith::subi(
                a.result(0).unwrap().into(),
                b.result(0).unwrap().into(),
                location_to_mlir(context, &expr.location),
            )
        },
        ExpressionBody::Mul(a, b) => {
            let a = expr_to_mlir(context, block, &a.read().unwrap());
            let b = expr_to_mlir(context, block, &b.read().unwrap());
            arith::muli(
                a.result(0).unwrap().into(),
                b.result(0).unwrap().into(),
                location_to_mlir(context, &expr.location),
            )
        },
        ExpressionBody::Div(a, b) => {
            let a = expr_to_mlir(context, block, &a.read().unwrap());
            let b = expr_to_mlir(context, block, &b.read().unwrap());
            arith::divsi(
                a.result(0).unwrap().into(),
                b.result(0).unwrap().into(),
                location_to_mlir(context, &expr.location),
            )
        },
        ExpressionBody::FunctionCall{fn_def, args} => {
            let name = &fn_def.name;
            match &fn_def.expr.upgrade().unwrap().read().unwrap().body {
                ExpressionBody::BuiltinFunction(f) => {
                    return builtin_function_call_to_mlir(context, block, expr);
                }
                ExpressionBody::Function{name, args, body, retern_type} => {
                    panic!("Not implemented. {:?}", fn_def);
                },
                _ => panic!("Not implemented. {:?}", fn_def),
            }
        },
        _ => panic!("Not implemented expr. {:?}", expr),
    };
    block.append_operation(operation)
}

fn type_to_mlir(context: &Context) -> Type<'_> {
    Type::index(&context)
}

fn location_to_mlir<'a>(context: &'a Context, location: &hir::Location) -> Location<'a> {
    Location::unknown(&context)
}

fn function_to_mlir<'a>(context: &'a Context, block: &'a Block<'a>, expr: &Expression) -> OperationRef<'a, 'a> {
    let ExpressionBody::Function{name, args, body, retern_type} = &expr.body else {
        panic!("Not a Function. {:?}", expr);
    };
    let i32_type = IntegerType::new(context, 32).into();
    let name_attr = StringAttribute::new(&context, name);
    let type_attr = TypeAttribute::new(
        FunctionType::new(&context, &[], &[i32_type]).into(),
    );
    let region = region_to_mlir(&context, body);
    let location = Location::unknown(&context);
    let operation = func::func(&context, name_attr, type_attr, region, &[], location);
    block.append_operation(operation)
}

fn builtin_function_to_mlir<'a>(context: &'a Context, block: &'a Block<'a>, expr: &Expression) -> OperationRef<'a, 'a> {
    let ExpressionBody::BuiltinFunction( name ) = &expr.body else {
        panic!("Not a builtinFunction. {:?}", expr);
    };
    let operation = match name {
        BuiltinFunction::Printf => {
            let i32_type = IntegerType::new(context, 32).into();
            let ptr_type = llvm::r#type::pointer(context, 0);
            llvm::func(
                context,
                StringAttribute::new(&context, "printf"),
                TypeAttribute::new(llvm::r#type::function(i32_type, &[ptr_type], true)),
                Region::new(),
                &[(
                    Identifier::new(&context, "linkage"),
                    llvm::attributes::linkage(&context, llvm::attributes::Linkage::External),
                )],
                location_to_mlir(context, &expr.location),
            )
        },
        _ => panic!("Not implemented. Built-in function: {:?}", name),
    };
    block.append_operation(operation)
}

fn builtin_function_call_to_mlir<'a>(context: &'a Context, block: &'a Block<'a>, expr: &Expression) -> OperationRef<'a, 'a> {
    let ExpressionBody::FunctionCall{fn_def, args} = &expr.body else {
        panic!("Not a FunctionCall. {:?}", expr);
    };
    let fn_def_expr = fn_def.expr.upgrade().unwrap();
    let ExpressionBody::BuiltinFunction(function) = &fn_def_expr.read().unwrap().body else {
        panic!("Not a builtinFunction. {:?}", expr);
    };
    let (fn_ty, ret_ty) = match function {
        BuiltinFunction::Printf => {
            let i32_type = IntegerType::new(context, 32).into();
            let ptr_type = llvm::r#type::pointer(context, 0);
            let fn_ty = llvm::r#type::function(i32_type, &[ptr_type], true);
            (fn_ty, i32_type)
        },
        _ => panic!("Not implemented. Built-in function: {:?}", function),
    };
    let operation = ods::llvm::CallOperationBuilder::new(context, location_to_mlir(context, &expr.location))
        .callee(FlatSymbolRefAttribute::new(&context, &fn_def.name))
        .var_callee_type(
            TypeAttribute::new(fn_ty),
        )
        .result(ret_ty)
        .callee_operands(&args
            .iter()
            .map(|expr| expr_to_mlir(context, block, &expr.read().unwrap()).result(0).unwrap().into())
            .collect::<Vec<_>>()
        )
        .build()
        .as_operation()
        .clone();
    block.append_operation(operation)
}

fn str_to_mlir<'a>(context: &'a Context, block: &'a Block<'a>, expr: &Expression) -> OperationRef<'a, 'a> {
    let ExpressionBody::Str{name, text} = &expr.body else {
        panic!("Not a Str. {:?}", expr);
    };
    let text = text.clone() + "\0";
    let size = text.len();
    let i8_type = IntegerType::new(&context, 8).into();
    let mem_ref_type = MemRefType::new(i8_type, &[size as i64], None, None);
    let operation = memref::global(
        context,
        name,
        Some("private"),
        mem_ref_type,
        Some(
            DenseElementsAttribute::new(
                RankedTensorType::new(&[size as u64], i8_type, None).into(),
                &text.bytes().map(|b| IntegerAttribute::new(i8_type, b as i64).into()).collect::<Vec<_>>(),
            )
            .unwrap()
            .into(),
        ),
        true,
        None,
        location_to_mlir(context, &expr.location),
    );
    block.append_operation(operation)
}
