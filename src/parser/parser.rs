#![allow(dead_code, unused)]

use std::borrow::Borrow;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{self, alphanumeric1, char, line_ending, multispace0, space0, space1};
use nom::combinator::{cond, eof, map, not, peek, value};
use nom::error::{context, ContextError, ParseError, VerboseError};
use nom::multi::{many0, many0_count};
use nom::sequence::{delimited, preceded, terminated, tuple};
use nom::IResult;

use crate::ast::{Ast, Expression, IndentedStatement, Statement};

pub fn parse(input: &str) -> SIResult<Ast> {
    map(
        terminated(
            many0(parse_line),
            tuple((multispace0, eof)),
        ),
        |v| Ast(v),
    )(input)
}

pub type SIResult<'a, T, E = VerboseError<&'a str>> = IResult<&'a str, T, E>;

pub fn parse_line(input: &str) -> SIResult<IndentedStatement> {
    context(
        "parse_line",
        map(
            preceded(
                many0(tuple((space0, line_ending))),
                tuple((space_indent, parse_statement, end_of_statemet)),
            ),
            |(indent, statement, _)| IndentedStatement(indent, statement),
        )
    )(input)
}

pub fn space_indent(input: &str) -> SIResult<usize> {
    many0_count(char(' '))(input)
}

pub fn end_of_statemet(input: &str) -> SIResult<()> {
    terminated(
        value((), space0),
        alt((
            value((), char(';')),
            value((), peek(char('}'))),
            value((), end_of_line_or_file),
        )),
    )(input)
}

pub fn end_of_line_or_file(input: &str) -> SIResult<(&str, &str)> {
    tuple((space0, alt((line_ending, eof))))(input)
}

pub fn parse_statement(input: &str) -> SIResult<Statement> {
    alt((
        map(parse_function, |expr| Statement::Function(expr)),
        map(parse_let, |expr| Statement::Let(expr)),
        map(parse_eval_expr, |expr| Statement::EvalExpr(expr)),
        map(parse_assign, |expr| Statement::Assign(expr)),
        map(parse_expr, |expr| Statement::Expr(expr)),
    ))(input)
}

pub fn parse_let(input: &str) -> SIResult<Expression> {
    map(
        tuple((keyword("let"), parse_expr, keyword("="), multispace0, parse_expr)),
        |(_, l, _, _, r)| Expression::Let(Box::new(l), Box::new(r))
    )(input)
}

pub fn parse_eval_expr(input: &str) -> SIResult<Expression> {
    map(
        tuple((keyword("="), parse_expr)),
        |(_, l)| l
    )(input)
}

pub fn parse_assign(input: &str) -> SIResult<Expression> {
    terminated(
        map(
            tuple((
                parse_term,
                bin_op("=", Expression::Assign),
            )),
            |(l, r)| l.reorder(r),
        ),
        space0,
    )(input)
}

pub fn parse_expr(input: &str) -> SIResult<Expression> {
    terminated(
        map(
            tuple((
                parse_term,
                many0(alt((
                    parse_add,  // expr + expr
                    parse_sub,  // expr - expr
                    parse_mul,  // expr * expr
                    parse_div,  // expr / expr
                )))
            )),
            |(l, r)| reorder(l, r),
        ),
        space0,
    )(input)
}

pub fn reorder(init: Expression, v: Vec<Expression>) -> Expression {
    let mut l = init;
    for e in v {
        l = l.reorder(e);
    }
    l
}

fn parse_function(input: &str) -> SIResult<Expression> {
    map(
        tuple((
            preceded(
                keyword("fn"),
                parse_identifier
            ),
            delimited(
                keyword("("),
                many0(parse_expr),
                keyword(")"),
            ),
            delimited(
                keyword("{"),
                many0(parse_line),
                keyword("}"),
            ),
        )),
        |(name, args, body)| Expression::Function {
            name: Box::new(name),
            args,
            body,
        }
    )(input)
}

pub fn parse_term(input: &str) -> SIResult<Expression> {
    terminated(alt((
        map(complete::i64, |i| Expression::I64(i)),
        parse_parentheses,    // ( expr )
        parse_identifier,    // alpha numeric string
    )), space0)(input)
}

pub fn parse_parentheses(input: &str) -> SIResult<Expression> {
    map(delimited(terminated(char('('), space0), parse_expr, char(')')), |expr| {
        Expression::Parentheses(Box::new(expr))
    })(input)
}

pub fn parse_identifier(input: &str) -> SIResult<Expression> {
    map(complete::alphanumeric1, |id| {
        Expression::Identifier(String::from(id))
    })(input)
}

pub fn keyword<'a>(t: &'a str) -> impl FnMut(&'a str) -> SIResult<'a, &'a str> {
    let is_a = alphanumeric1::<_, ()>(t).is_ok();
    terminated(tag(t), alt((
        value((), space1),
        value((), peek(tuple((
            cond(is_a, not(alphanumeric1)),
            // cond(!is_a, alphanumeric1),
        )))),
    )))
}

pub fn bin_op<'a>(
    op: &'a str,
    f: impl 'a + Fn(Box<Expression>, Box<Expression>) -> Expression,
) -> impl FnMut(&'a str) -> SIResult<'a, Expression> {
    map(
        tuple((
            delimited(
                multispace0,
                tag(op),
                multispace0,
            ),
            parse_term,
        )),
        move |(op, r)| f(Box::new(Expression::Resolving()), Box::new(r))
    )
}

pub fn parse_add(input: &str) -> SIResult<Expression> {
    bin_op("+", Expression::Add)(input)
    // bin_op(
    //     "+",
    //     |l, r| Expression::Add(l, r)
    // )(input)
    // map(tuple((
    //     terminated(tag("+"), space0),
    //     parse_term,
    // )), |(op, t)| Expression::Add(Box::new(Expression::None()), Box::new(t)))(input)
}

pub fn parse_sub(input: &str) -> SIResult<Expression> {
    bin_op("-", Expression::Sub)(input)
}

pub fn parse_mul(input: &str) -> SIResult<Expression> {
    bin_op("*", Expression::Mul)(input)
}

pub fn parse_div(input: &str) -> SIResult<Expression> {
    bin_op("/", Expression::Div)(input)
}

// pub fn parse_multiplication(input: &str) -> SIResult<Expression> {
//     // bin_op(
//     //     "*",
//     //     |l, r| Expression::Multiplication(l, r)
//     // )(input)
//     map(tuple((parse_expr, char('*'), parse_expr)), |(l, op, r)| {
//         Expression::Mul(Box::new(l), Box::new(r))
//     })(input)
// }


#[cfg(test)]
mod tests {
    use nom::error::Error;
    use nom::{error::convert_error, Err, Finish};
    use nom::combinator::{eof, map, opt, value};

    use super::*;

    #[test]
    fn empty_line() {
        assert!(eof::<&str, Error<&str>>("").finish().is_ok());
        // assert!(parse_line("").finish().is_ok());
        assert!(parse("").finish().is_ok());
    }

    #[test]
    fn test() {
        // unsafe { backtrace_on_stack_overflow::enable() };
        // let input = "  \nlet\nabc";
        let input = "\n10 + 1 \n + (2) \n 1 + a2\n\n";
        println!("input = {}", input);
        let result = parse(input).finish();
        println!("result = {:?}", result);
        if let Err(e) = &result {
            println!("convert_error: \n{}", convert_error(input, e.clone()));
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_end_of_line_or_file() {
        let input = "";
        println!("input = {}", input);
        let result = end_of_line_or_file(input).finish();
        println!("result = {:?}", result);
        if let Err(e) = &result {
            println!("convert_error: \n{}", convert_error(input, e.clone()));
        }
        assert!(result.is_ok());
    }

    // fn printErr() {

    // }
}
