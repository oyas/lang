use std::sync::{Arc, RwLock};

use super::{BasicType, Block, BuiltinFunction, Definition, Expression, ExpressionBody, Hir, Module, Region, Type};


pub fn pass_in_hir(hir: &mut Hir) {
    pass_in_region(&hir.read().body);
}

pub fn pass_in_region(region: &Arc<RwLock<Region>>) {
    region.read().unwrap().blocks.iter().for_each(|b| pass_in_block(&b));
}

pub fn pass_in_block(block: &Block) {
    block.exprs.iter().for_each(|expr| {
        pass_in_expr(expr);
    });
}

pub fn pass_in_expr(expr: &Arc<RwLock<Expression>>) {
    let module = expr.read().unwrap().region.upgrade().unwrap().read().unwrap().module.as_ref().unwrap().upgrade().unwrap();
    let binding = expr.write().unwrap();
    match &binding.body {
        ExpressionBody::Str{..}
        | ExpressionBody::I64{..}
        | ExpressionBody::Identifier(_)
        => (),
        ExpressionBody::Add(a, b)
        | ExpressionBody::Sub(a, b)
        | ExpressionBody::Mul(a, b)
        | ExpressionBody::Div(a, b)
        | ExpressionBody::Assign(a, b)
        => {
            pass_in_expr(a);
            pass_in_expr(b);
        },
        ExpressionBody::BuiltinFunction(f) => {
            match &f {
                BuiltinFunction::Printf => {
                }
            }
        },
        ExpressionBody::Function{name, args, body, retern_type} => {
            args.iter().for_each(|a| pass_in_expr(a));
            body.write().unwrap().blocks.iter().for_each(|b| pass_in_block(b));
        },
        ExpressionBody::FunctionCall{fn_def, args} => {
            match fn_def.expr.upgrade().unwrap().read().unwrap().body {
                ExpressionBody::BuiltinFunction(BuiltinFunction::Printf) => {
                    let s = match args[1].read().unwrap().ty {
                        Type::BasicType(BasicType::I64) => "%d\n",
                        Type::BasicType(BasicType::Str) => "%s\n",
                        _ => panic!("Not implemented for print. {:?}", args[1].read().unwrap()),
                    };
                    let fmt = const_string(&module, s);
                    args[0].write().unwrap().body = fmt.expr.upgrade().unwrap().read().unwrap().body.clone();
                }
                _ => panic!("Not implemented. {:?}", fn_def),
            }
        },
        ExpressionBody::Let(_, b) => pass_in_expr(b),
        ExpressionBody::Variable(_) => todo!(),
        // ExpressionBody::Parentheses(a) => pass_in_expr(hir, a),
        ExpressionBody::None() => todo!(),
        _ => panic!("Not implemented {:?}", expr),
    };
}

fn const_string(module: &Arc<RwLock<Module>>, s: &str) -> Definition {
    let region = Arc::clone(&module.read().unwrap().pre_body);
    let name = format!(".str_{}", module.read().unwrap().module_symbols.len());
    let e = ExpressionBody::Str{name: name.to_string(), text: s.to_string()}.to_expression(&region);
    let d = Definition::new(false, &name, &e, &region);
    region.write().unwrap().blocks[0].exprs.push(e);
    region.write().unwrap().symbols.insert(name, d.clone());
    d
}
