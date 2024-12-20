use std::{any::Any, error::Error, ops::Deref};

use inkwell::values::{BasicValue, IntValue};

use crate::ast::Expression;

use super::CodeGen;


pub fn build(codegen: &CodeGen, expr: &Expression) -> Result<(), Box<dyn Error>> {
    let ret = match expr {
        Expression::Add(a, b) => {
            let Expression::I64(ai) = **a else { panic!("'a' is not i64.") };
            let Expression::I64(bi) = **b else { panic!("'b' is not i64.") };
            let i32_type = codegen.context.i32_type();
            let lhs = i32_type.const_int(ai as u64, false);
            let rhs = i32_type.const_int(bi as u64, false);
            codegen.builder.build_int_add(lhs, rhs, "add").unwrap()
        },
        _ => {
            panic!("not implemented");
        },
    };

    codegen.builder.build_return(Some(&ret)).unwrap();

    Ok(())
}

pub fn build_expression<'a>(codegen: &CodeGen<'a>, expr: &Expression) -> Result<IntValue<'a>, Box<dyn Error>> {
    let ret = match expr {
        Expression::I64(a) => {
            let i32_type = codegen.context.i32_type();
            i32_type.const_int(*a as u64, false)
        },
        Expression::Identifier(a) => {
            codegen.values.borrow().get(a).unwrap().as_basic_value_enum().into_int_value()
        },
        Expression::Add(a, b) => {
            let lhs = build_expression(codegen, a).unwrap();
            let rhs = build_expression(codegen, b).unwrap();
            codegen.builder.build_int_add(lhs, rhs, "add").unwrap()
        },
        Expression::Sub(a, b) => {
            let lhs = build_expression(codegen, a).unwrap();
            let rhs = build_expression(codegen, b).unwrap();
            codegen.builder.build_int_sub(lhs, rhs, "sub").unwrap()
        },
        Expression::Mul(a, b) => {
            let lhs = build_expression(codegen, a).unwrap();
            let rhs = build_expression(codegen, b).unwrap();
            codegen.builder.build_int_mul(lhs, rhs, "mul").unwrap()
        },
        Expression::Div(a, b) => {
            let lhs = build_expression(codegen, a).unwrap();
            let rhs = build_expression(codegen, b).unwrap();
            codegen.builder.build_int_signed_div(lhs, rhs, "div").unwrap()
        },
        Expression::Parentheses(a) => build_expression(codegen, a).unwrap(),
        Expression::Let(a, b) => {
            let Expression::Identifier(ref l) = **a else {
                panic!("Cannot assign to non-identifier.")
            };
            if codegen.values.borrow().contains_key(l) {
                panic!("Variable already exists.")
            }
            let r = build_expression(codegen, b).unwrap();
            codegen.values.borrow_mut().insert(l.clone(), r.as_basic_value_enum());
            codegen.context.i32_type().const_zero()
        },
        Expression::Assign(a, b) => {
            let Expression::Identifier(ref l) = **a else {
                panic!("Cannot assign to non-identifier.")
            };
            if !codegen.values.borrow().contains_key(l) {
                panic!("Variable does not exist.")
            }
            let r = build_expression(codegen, b).unwrap();
            codegen.values.borrow_mut().insert(l.clone(), r.as_basic_value_enum());
            codegen.context.i32_type().const_zero()
        },
        _ => {
            panic!("not implemented");
        },
    };
    Ok(ret)
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, execution_engine::JitFunction};

    use crate::backend::llvm::{codegen, link, target};

    use super::*;

    #[test]
    fn test() {
        let context = Context::create();
        let codegen = codegen::new(&context).unwrap();
        let module = codegen.get_main_module();

        let i32_type = codegen.context.i32_type();
        let fn_type = i32_type.fn_type(&[], false);
        let main_function = module.add_function("main", fn_type, None);
        let entry_block = codegen.context.append_basic_block(main_function, "entry");
        codegen.builder.position_at_end(entry_block);

        let expr = Expression::Add(
            Box::new(Expression::I64(1)),
            Box::new(Expression::I64(3)),
        );

        build(&codegen, &expr).unwrap();

        // show ll
        println!("----- Generated LLVM IR -----");
        println!("{}", module.to_string());
        println!("----- End of LLVM IR -----");

        // JIT
        type MainFunc = unsafe extern "C" fn() -> i32;
        unsafe {
            let main: JitFunction<MainFunc> = codegen.execution_engine.get_function("main").unwrap();
            let result = main.call();
            println!("result = {:?}", result);
        };
    }
}
