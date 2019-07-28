#![allow(dead_code)]

use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fmt;


#[derive(Debug)]
pub struct Element {
    pub value: Value,
    pub value_type: ValueType,
    pub childlen: Vec<Element>,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub enum Value {
    Identifier(String),     // start with alphabetic char; variable or
    Integer(i64),           // 1234
    String(String),
    Operator(String,i32),   // infix notation, whitch required left and right element (operator string, priority)
    Formula(String),
    Scope(Box<Value>),
    Brackets(String),
    EndLine(),              // "\n"
    FileScope(),            // the top level element
}

pub fn get_element(token: &str) -> Option<Element> {
    match token.chars().nth(0) {
        Some(n) if n.is_ascii_digit() => {
            match token.parse::<i64>() {
                Ok(value) =>  {
                    Some(Element{
                        value: Value::Integer(value),
                        value_type: ValueType::Inference,
                        childlen: Vec::new(),
                    })
                }
                Err(err) => {
                    panic!("can't parse \"{}\" as integer: {}", token, err);
                }
            }
        }
        Some('+') | Some('-') | Some('*') | Some('/') => {
            Some(Element{
                value: get_operator(token),
                value_type: ValueType::Inference,
                childlen: Vec::new(),
            })
        }
        Some('(') | Some(')') => {
            Some(Element{
                value: Value::Brackets(token.to_string()),
                value_type: ValueType::Inference,
                childlen: Vec::new(),
            })
        }
        Some('\n') => {
            Some(Element{
                value: Value::EndLine(),
                value_type: ValueType::None,
                childlen: Vec::new(),
            })
        }
        _ => None,
    }
}

lazy_static! {
    pub static ref OPERATORS: HashMap<String, Value> = {
        let mut m = HashMap::new();
        m.insert("*".to_string(), Value::Operator("*".to_string(), 20));
        m.insert("/".to_string(), Value::Operator("/".to_string(), 20));
        m.insert("+".to_string(), Value::Operator("+".to_string(), 10));
        m.insert("-".to_string(), Value::Operator("-".to_string(), 10));
        m
    };
}

pub fn get_operator(ope: &str) -> Value {
    return OPERATORS[ope].clone()
}

pub fn get_end_bracket(bra: &str) -> Option<String> {
    match bra {
        "(" => Some(")".to_string()),
        "{" => Some("}".to_string()),
        _ => None,
    }
}

#[derive(Debug)]
pub enum ValueType {
    Inference,
    None,
}


impl Element {
    fn print_with_prefix(&self, fmt: &mut fmt::Formatter, prefix: &str) -> fmt::Result {
        if let Err(err) = fmt.write_fmt(format_args!("{}{:?} : {:?}", prefix, self.value, self.value_type)) {
            return Err(err);
        }
        for child in &self.childlen {
            fmt.write_str("\n").ok();
            &child.print_with_prefix(fmt, &format!("{}. ", prefix));
        }
        Ok(())
    }
}

impl fmt::Display for Element {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.print_with_prefix(fmt, "")
    }
}
