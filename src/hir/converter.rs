use std::sync::{Arc, RwLock};

use crate::ast::{self, Ast, Statement};

use super::{Block, BuiltinFunction, Definition, Expression, ExpressionBody, Hir, Module, Region, Terminator, Type};

pub fn convert_from_ast(ast: &Ast, filename: &str) -> Hir {
    let hir = Hir::new(filename, filename);
    let region = hir.read().body.clone();
    for ast::IndentedStatement(_a, b) in &ast.0 {
        let e = statement_to_hir(&region, b);
        hir.read().body.write().unwrap().blocks[0].exprs.push(e);
    }
    hir
}

fn statement_to_hir(region: &Arc<RwLock<Region>>, statement: &Statement) -> Arc<RwLock<Expression>> {
    match &statement {
        Statement::Let(expr) => {
            let ast::Expression::Let(a, b) = expr else {
                panic!("This is not let.");
            };
            let ast::Expression::Identifier(ref name) = **a else {
                panic!("This is not Identifier.");
            };
            let e = ast_expr_to_hir(region, b);
            let l = ExpressionBody::Let(name.clone(), e).to_expression(region);
            let d = Definition::new(true, &name, &l, region);
            region.write().unwrap().symbols.insert(name.clone(), d);
            l
        }
        Statement::Assign(expr) => {
            if let ast::Expression::Assign(a, b) = expr {
                ExpressionBody::None().to_expression(region)
            } else {
                panic!("This is not assign.");
            }
        },
        Statement::EvalExpr(expr) => {
            let name = region.read().unwrap().module.as_ref().unwrap().upgrade().unwrap().read().unwrap().module_name.clone();
            let body = Arc::new(RwLock::new(Region::new(region)));
            let mut block = Block::new(&body);
            let e1 = ast_expr_to_hir(region, expr);
            let print_ret = print_expr(region, &e1);
            block.exprs.push(e1);
            block.exprs.push(print_ret);
            block.terminator = Some(Terminator::Return);
            body.write().unwrap().blocks.push(block);
            let e = ExpressionBody::Function{
                name: name.clone(),
                args: Vec::new(),
                body,
                retern_type: Type::Resolving(),
            }.to_expression(region);
            let d = Definition::new(true, &name, &e, region);
            region.write().unwrap().symbols.insert(name.clone(), d);
            e
        },
        Statement::Expr(expr) => {
            ast_expr_to_hir(region, expr)
        },
        Statement::Function(expr) => {
            let ast::Expression::Function{name, args, body} = expr else {
                panic!("This is not Function.");
            };
            let ast::Expression::Identifier(ref name) = **name else {
                panic!("This is not Identifier.");
            };
            let e = ExpressionBody::Function{
                name: name.clone(),
                args: args.iter().map(|x| ast_expr_to_hir(region, x)).collect(),
                body: Arc::new(RwLock::new(Region::new(region))),
                retern_type: Type::Resolving(),
            }.to_expression(region);
            let d = Definition::new(true, &name, &e, region);
            region.write().unwrap().symbols.insert(name.clone(), d);
            e
        },
    }
}

fn ast_expr_to_hir(region: &Arc<RwLock<Region>>, expression: &ast::Expression) -> Arc<RwLock<Expression>> {
    match expression {
        ast::Expression::None() => ExpressionBody::None().to_expression(region),
        ast::Expression::Identifier(name) => {
            ExpressionBody::Identifier(name.clone()).to_expression(region)
        },
        // ast::Expression::Typed(a, b) => {
        //     let a = ast_expr_to_hir(a);
        //     a.ty = b.clone();
        //     a
        // },
        ast::Expression::I64(i) => {
            ExpressionBody::I64(*i).to_expression(region)
        },
        ast::Expression::Add(a, b) => {
            let a = ast_expr_to_hir(region, a);
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Add(a, b).to_expression(region)
        },
        ast::Expression::Sub(a, b) => {
            let a = ast_expr_to_hir(region, a);
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Sub(a, b).to_expression(region)
        },
        ast::Expression::Mul(a, b) => {
            let a = ast_expr_to_hir(region, a);
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Mul(a, b).to_expression(region)
        },
        ast::Expression::Div(a, b) => {
            let a = ast_expr_to_hir(region, a);
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Div(a, b).to_expression(region)
        },
        ast::Expression::Parentheses(a) => {
            ast_expr_to_hir(region, a)
            // Arc::new(RwLock::new(ExpressionBody::Parentheses(a).to_expression()))
        },
        ast::Expression::Let(a, b) => {
            let ast::Expression::Identifier(ref name) = **a else {
                panic!("This is not Identifier.");
            };
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Let(name.clone(), b).to_expression(region)
        },
        ast::Expression::Assign(a, b) => {
            let a = ast_expr_to_hir(region, a);
            let b = ast_expr_to_hir(region, b);
            ExpressionBody::Assign(a, b).to_expression(region)
        },
        _ => panic!("Not implemented."),
    }
}

fn print_expr(region: &Arc<RwLock<Region>>, e: &Arc<RwLock<Expression>>) -> Arc<RwLock<Expression>> {
    let fn_printf = define_printf(&region.write().unwrap().module.as_ref().unwrap().upgrade().unwrap());
    // let s = match e.read().unwrap().ty {
    //     Type::BasicType(BasicType::I64) => "%d\n",
    //     Type::BasicType(BasicType::Str) => "%s\n",
    //     _ => panic!("Not implemented for print. {:?}", e),
    // };
    // let fmt = const_string(hir, s);
    // Arc::new(RwLock::new(ExpressionBody::FunctionCall{
    //     fn_def: fn_printf,
    //     args: vec![fmt.expr.upgrade().unwrap().clone(), e.clone()],
    // }.to_expression()))
    let fmt_expr = ExpressionBody::Str{name: String::new(), text: String::new()}.to_expression(region);
    ExpressionBody::FunctionCall{
        fn_def: fn_printf,
        // args: vec![fmt.expr.upgrade().unwrap().clone(), e.clone()],
        args: vec![fmt_expr, e.clone()],
    }.to_expression(region)
}

fn define_printf(module: &Arc<RwLock<Module>>) -> Definition {
    let name = "printf".to_string();
    let region = &module.read().unwrap().pre_body;
    let e = BuiltinFunction::Printf.to_expression(region);
    let d = Definition::new(false, &name, &e, region);
    region.write().unwrap().blocks[0].exprs.push(e);
    region.write().unwrap().symbols.insert(name.clone(), d.clone());
    d
}
