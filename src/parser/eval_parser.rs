#![allow(dead_code, unused)]

use nom::branch::alt;
use nom::character::complete::{self, alphanumeric1, char, line_ending, multispace0, none_of, one_of, space0, space1};
use nom::combinator::{cond, eof, map, not, peek, value};
use nom::multi::{many0, many1};
use nom::sequence::tuple;
use nom::IResult;

use super::{end_of_line_or_file, SIResult};

// check if the input does not have incomplete parenthesis/brackets/braces
pub fn is_complete(input: &str) -> SIResult<()> {
    value((), tuple((
        many0(complete_term),
        eof,
    )))
    (input)
}

fn complete_term(input: &str) -> SIResult<()> {
    alt((
        value((), line_ending),
        value((), none_of("(){}[]=")),
        value((), tuple((char('('), many0(complete_term), char(')')))),
        value((), tuple((char('{'), many0(complete_term), char('}')))),
        value((), tuple((char('['), many0(complete_term), char(']')))),
        value((), tuple((char('='), many0(end_of_line_or_file), complete_term))),
    ))(input)
}

#[cfg(test)]
mod tests {
    use nom::error::Error;
    use nom::{error::convert_error, Err, Finish};
    use nom::combinator::{eof, map, opt, value};

    use super::*;

    #[test]
    fn empty_line() {
        test_debug("", true)
    }

    #[test]
    fn test() {
        test_debug("1", true);
        test_debug("1 + 2\n3 * 4", true);
        test_debug("\n10 + 1 \n + (2) \n 1 + a2\n\n", true);
        test_debug("\n10 + 1 \n + (2 \n 1 + a2\n\n", false);
        test_debug("()[]{}", true);
        test_debug("(\n)[\n]{\n}", true);
        test_debug("(\n)[\n]{\n", false);
        test_debug("let a =", false);
        test_debug("let a =\n\n", false);
        test_debug("let a =\n    10", true);
        test_debug("let a =\n\n(2\n)", true);
        test_debug("let a =\n\n1 + (2\n", false);
        test_debug("fn f(", false);
        test_debug("fn f(\na: String\n", false);
        test_debug("fn f(\na: String\n) -> String {", false);
        test_debug("fn f(\na: String\n) -> String {\n  let b = a\n", false);
        test_debug("fn f(\na: String\n) -> String {\n  let b = a\n}", true);
    }

    fn test_debug(input: &str, expected: bool) {
        println!("input = {}", input);
        let result = is_complete(input);
        println!("result = {:?}", result);
        if let Err(e) = &result {
            if let Err::Error(e) = e {
                println!("convert_error: \n{}", convert_error(input, e.clone()));
            }
        }
        if expected {
            assert!(result.is_ok());
        } else {
            assert!(result.is_err());
        }
    }
}
