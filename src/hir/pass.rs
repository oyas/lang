use std::sync::{Arc, RwLock};

use super::{BasicType, Block, BuiltinFunction, Definition, Expression, ExpressionBody, Hir, Region, Type};


pub fn pass_in_hir(hir: &mut Hir) {
    let mut hir2 = Hir::new("", "");
    pass_in_region(&mut hir2, &hir.body);
    hir.body.blocks[0].exprs = [hir2.body.blocks[0].exprs.clone(), hir.body.blocks[0].exprs.clone()].concat();
    hir.module_symbols.extend(hir2.module_symbols);
}

pub fn pass_in_region(hir: &mut Hir, region: &Region) {
    region.blocks.iter().for_each(|b| pass_in_block(hir, &b));
}

pub fn pass_in_block(hir: &mut Hir, block: &Block) {
    block.exprs.iter().for_each(|expr| {
        pass_in_expr(hir, expr);
    });
}

pub fn pass_in_expr(hir: &mut Hir, expr: &Arc<RwLock<Expression>>) {
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
        | ExpressionBody::Let(a, b)
        | ExpressionBody::Assign(a, b)
        => {
            pass_in_expr(hir, a);
            pass_in_expr(hir, b);
        },
        ExpressionBody::BuiltinFunction(f) => {
            match &f {
                BuiltinFunction::Printf => {
                }
            }
        },
        ExpressionBody::Function{name, args, body, retern_type} => {
            args.iter().for_each(|a| pass_in_expr(hir, a));
            body.blocks.iter().for_each(|b| pass_in_block(hir, b));
        },
        ExpressionBody::FunctionCall{fn_def, args} => {
            match fn_def.expr.upgrade().unwrap().read().unwrap().body {
                ExpressionBody::BuiltinFunction(BuiltinFunction::Printf) => {
                    let s = match args[1].read().unwrap().ty {
                        Type::BasicType(BasicType::I64) => "%d\n",
                        Type::BasicType(BasicType::Str) => "%s\n",
                        _ => panic!("Not implemented for print. {:?}", args[1].read().unwrap()),
                    };
                    let fmt = const_string(hir, s);
                    args[0].write().unwrap().body = fmt.expr.upgrade().unwrap().read().unwrap().body.clone();
                }
                _ => panic!("Not implemented. {:?}", fn_def),
            }

        },
        ExpressionBody::Variable(_) => todo!(),
        ExpressionBody::Parentheses(a) => pass_in_expr(hir, a),
        ExpressionBody::None() => todo!(),
        _ => panic!("Not implemented {:?}", expr),
    };
}

fn const_string(hir: &mut Hir, s: &str) -> Definition {
    let name = format!(".str_{}", hir.module_symbols.len());
    let e = Arc::new(RwLock::new(ExpressionBody::Str{name: name.to_string(), text: s.to_string()}.to_expression()));
    let d = Definition::new(false, &name, Arc::downgrade(&e));
    hir.body.blocks[0].exprs.push(e);
    hir.module_symbols.insert(name.clone(), d.clone());
    d
}
