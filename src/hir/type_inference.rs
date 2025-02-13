use std::sync::{Arc, RwLock};

use super::{BasicType, Block, Expression, ExpressionBody, Hir, Region, Type};


pub fn inference_in_hir(hir: &mut Hir) {
    inference_in_region(&mut hir.body);
}

pub fn inference_in_region(region: &Region) {
    region.blocks.iter().for_each(|b| inference_in_block(&b));
}

pub fn inference_in_block(block: &Block) {
    block.exprs.iter().for_each(|expr| {
        inference_in_expr(expr);
    });
}

pub fn inference_in_expr(expr: &Arc<RwLock<Expression>>) -> Type {
    let ty = match &expr.read().unwrap().body {
        ExpressionBody::Str{..} => Type::BasicType(BasicType::Str),
        ExpressionBody::I64{..} => Type::BasicType(BasicType::I64),
        ExpressionBody::Add(a, b)
        | ExpressionBody::Sub(a, b)
        | ExpressionBody::Mul(a, b)
        | ExpressionBody::Div(a, b)
        => {
            let a_ty = inference_in_expr(a);
            let b_ty = inference_in_expr(b);
            if a_ty != b_ty {
                panic!("Type mismatch");
            }
            a_ty.clone()
        },
        // ExpressionBody::Parentheses(a) => inference_in_expr(a),
        ExpressionBody::Function{name, args, body, retern_type} => {
            let args_ty = args.iter().map(|a| inference_in_expr(a)).collect();
            body.blocks.iter().for_each(|b| inference_in_block(b));
            Type::Function(args_ty, vec![retern_type.clone()])
        },
        ExpressionBody::BuiltinFunction{..} => Type::Resolving(),
        ExpressionBody::FunctionCall{..} => Type::Resolving(),
        _ => panic!("Not implemented {:?}", expr),
    };
    expr.write().unwrap().ty = ty.clone();
    ty
}
