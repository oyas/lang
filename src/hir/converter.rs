use std::sync::{Arc, RwLock};

use crate::ast::{self, Ast, Statement};

use super::{BasicType, Block, BuiltinFunction, Definition, Expression, ExpressionBody, Hir, Location, Region, Terminator, Type};

pub fn convert_from_ast(ast: &Ast, filename: &str) -> Hir {
    let mut hir = Hir::new(filename, filename);
    for ast::IndentedStatement(_a, b) in &ast.0 {
        let e = statement_to_hir(&mut hir, b);
        hir.body.blocks[0].exprs.push(e);
    }
    hir
}

fn statement_to_hir(hir: &mut Hir, statement: &Statement) -> Arc<RwLock<Expression>> {
    match &statement {
        Statement::Let(expr) => {
            if let ast::Expression::Let(a, b) = expr {
                let ast::Expression::Identifier(ref name) = **a else {
                    panic!("This is not Identifier.");
                };
                let e = Arc::new(RwLock::new(ExpressionBody::None().to_expression()));
                let d = Definition::new(true, &name, Arc::downgrade(&e));
                hir.body.symbols.insert(name.clone(), d);
                e
            } else {
                panic!("This is not let.");
            }
        }
        Statement::Assign(expr) => {
            if let ast::Expression::Assign(a, b) = expr {
                Arc::new(RwLock::new(ExpressionBody::None().to_expression()))
            } else {
                panic!("This is not assign.");
            }
        },
        Statement::EvalExpr(expr) => {
            let name = hir.module_name.clone();
            let mut body = Region::new();
            let mut block = Block::new();
            let e1 = ast_expr_to_hir(expr);
            let print_ret = print_expr(hir, &e1);
            block.exprs.push(e1);
            block.exprs.push(print_ret);
            block.terminator = Some(Terminator::Return);
            body.blocks.push(block);
            let e = Arc::new(RwLock::new(ExpressionBody::Function{
                name: name.clone(),
                args: Vec::new(),
                body,
                retern_type: Type::Resolving(),
            }.to_expression()));
            let d = Definition::new(true, &name, Arc::downgrade(&e));
            hir.body.symbols.insert(name.clone(), d);
            e
        },
        Statement::Expr(expr) => {
            ast_expr_to_hir(expr)
        },
        Statement::Function(expr) => {
            if let ast::Expression::Function{name, args, body} = expr {
                let ast::Expression::Identifier(ref name) = **name else {
                    panic!("This is not Identifier.");
                };
                let e = Arc::new(RwLock::new(ExpressionBody::Function{
                    name: name.clone(),
                    args: args.iter().map(|x| ast_expr_to_hir(x)).collect(),
                    body: Region::new(),
                    retern_type: Type::Resolving(),
                }.to_expression()));
                let d = Definition::new(true, &name, Arc::downgrade(&e));
                hir.body.symbols.insert(name.clone(), d);
                e
            } else {
                panic!("This is not Function.");
            }
        },
    }
}

fn ast_expr_to_hir(expression: &ast::Expression) -> Arc<RwLock<Expression>> {
    match expression {
        ast::Expression::None() => Arc::new(RwLock::new(ExpressionBody::None().to_expression())),
        ast::Expression::Identifier(name) => {
            Arc::new(RwLock::new(ExpressionBody::Identifier(name.clone()).to_expression()))
        },
        // ast::Expression::Typed(a, b) => {
        //     let a = ast_expr_to_hir(a);
        //     a.ty = b.clone();
        //     a
        // },
        ast::Expression::I64(i) => {
            Arc::new(RwLock::new(ExpressionBody::I64(*i).to_expression()))
        },
        ast::Expression::Add(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Add(a, b).to_expression()))
        },
        ast::Expression::Sub(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Sub(a, b).to_expression()))
        },
        ast::Expression::Mul(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Mul(a, b).to_expression()))
        },
        ast::Expression::Div(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Div(a, b).to_expression()))
        },
        ast::Expression::Parentheses(a) => {
            let a = ast_expr_to_hir(a);
            Arc::new(RwLock::new(ExpressionBody::Parentheses(a).to_expression()))
        },
        ast::Expression::Let(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Let(a, b).to_expression()))
        },
        ast::Expression::Assign(a, b) => {
            let a = ast_expr_to_hir(a);
            let b = ast_expr_to_hir(b);
            Arc::new(RwLock::new(ExpressionBody::Assign(a, b).to_expression()))
        },
        _ => panic!("Not implemented."),
    }
}

fn print_expr(hir: &mut Hir, e: &Arc<RwLock<Expression>>) -> Arc<RwLock<Expression>> {
    let fn_printf = define_printf(hir);
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
    let fmt_expr = Arc::new(RwLock::new(ExpressionBody::Str{name: String::new(), text: String::new()}.to_expression()));
    Arc::new(RwLock::new(ExpressionBody::FunctionCall{
        fn_def: fn_printf,
        // args: vec![fmt.expr.upgrade().unwrap().clone(), e.clone()],
        args: vec![fmt_expr, e.clone()],
    }.to_expression()))
}

fn define_printf(hir: &mut Hir) -> Definition {
    let name = "printf".to_string();
    let e = Arc::new(RwLock::new(BuiltinFunction::Printf.to_expression()));
    let d = Definition::new(false, &name, Arc::downgrade(&e));
    hir.body.blocks[0].exprs.push(e);
    hir.module_symbols.insert(name.clone(), d.clone());
    d
}

fn const_string(hir: &mut Hir, s: &str) -> Definition {
    let name = format!(".str_{}", hir.module_symbols.len());
    let e = Arc::new(RwLock::new(ExpressionBody::Str{name: name.to_string(), text: s.to_string()}.to_expression()));
    let d = Definition::new(false, &name, Arc::downgrade(&e));
    hir.body.blocks[0].exprs.push(e);
    hir.module_symbols.insert(name.clone(), d.clone());
    d
}
