#![allow(dead_code)]

use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Element {
    pub value: Value,
    pub value_type: ValueType,
    pub childlen: Vec<Element>,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub enum Value {
    Identifier(String), // start with alphabetic char; variable or
    Integer(i64),       // 1234
    String(String),
    Boolean(bool),              // true or false
    Operator(String, i32), // infix notation, whitch required left and right element (operator string, priority)
    UnaryOperator(String, i32), // "-", "!"
    Symbol(String),        // reserved words: let, if, for, func ...
    Formula(String),
    Scope(),
    Bracket(String), // "(", ")", "{", "}", ...
    EndLine(),       // "\n"
    FileScope(),     // the top level element
    Type(i64),
    Space(i64), // space or indent. value is count of spaces. But, if include tabs, the value is -1
}

pub fn get_element(token: &str) -> Option<Element> {
    match token.chars().nth(0) {
        Some(n) if n.is_ascii_digit() => match token.parse::<i64>() {
            Ok(value) => Some(Element {
                value: Value::Integer(value),
                value_type: ValueType::Inference,
                childlen: Vec::new(),
            }),
            Err(err) => {
                panic!("can't parse \"{}\" as integer: {}", token, err);
            }
        },
        Some(a) if a.is_ascii_alphabetic() => Some(Element {
            value: get_identifier(token),
            value_type: ValueType::Inference,
            childlen: Vec::new(),
        }),
        Some('+') | Some('-') | Some('*') | Some('/') | Some('=') | Some('!') | Some('(')
        | Some(')') | Some('{') | Some('}') => {
            let value = get_identifier(token);
            match value {
                Value::Operator(..) | Value::UnaryOperator(..) | Value::Bracket(..) => {
                    Some(Element {
                        value: value,
                        value_type: ValueType::Inference,
                        childlen: Vec::new(),
                    })
                }
                _ => None,
            }
        }
        Some('\n') => Some(Element {
            value: Value::EndLine(),
            value_type: ValueType::None,
            childlen: Vec::new(),
        }),
        Some(' ') => {
            let mut count = 0;
            for c in token.chars() {
                if c == ' ' {
                    count += 1;
                } else {
                    count = -1;
                    break;
                }
            }
            Some(Element {
                value: Value::Space(count),
                value_type: ValueType::None,
                childlen: Vec::new(),
            })
        }
        _ => None,
    }
}

pub fn get_next_operator(tokens: &[String], mut pos: usize) -> Option<Element> {
    while pos < tokens.len() {
        if let Some(el) = get_element(&tokens[pos]) {
            match el.value {
                Value::EndLine() | Value::Space(_) => {}
                Value::Integer(..) => return None,
                _ => {
                    if pos + 1 < tokens.len() {
                        if let Some(el) =
                            get_element(&format!("{}{}", &tokens[pos], &tokens[pos + 1]))
                        {
                            if let Value::Operator(..) = el.value {
                                return Some(el);
                            }
                        }
                    }
                    if let Some(el) = get_element(&tokens[pos]) {
                        if let Value::Operator(..) = el.value {
                            return Some(el);
                        }
                    }
                    return None;
                }
            }
        }
        pos += 1;
    }
    None
}

pub fn get_next_nonblank_element(tokens: &[String], mut pos: usize) -> Option<Element> {
    while pos < tokens.len() {
        if let Some(el) = get_element(&tokens[pos]) {
            match el.value {
                Value::EndLine() | Value::Space(_) => {}
                _ => return Some(el),
            }
        }
        pos += 1;
    }
    None
}

lazy_static! {
    pub static ref SYMBOLS: HashMap<String, Value> = {
        let mut m = HashMap::new();
        // Operators
        m.insert("!".to_string(), Value::UnaryOperator("!".to_string(), 50));
        m.insert("*".to_string(), Value::Operator("*".to_string(), 30));
        m.insert("/".to_string(), Value::Operator("/".to_string(), 30));
        m.insert("+".to_string(), Value::Operator("+".to_string(), 20));
        m.insert("-".to_string(), Value::Operator("-".to_string(), 20));
        m.insert("==".to_string(), Value::Operator("==".to_string(), 15));
        m.insert("!=".to_string(), Value::Operator("!=".to_string(), 15));
        m.insert("=".to_string(), Value::Operator("=".to_string(), 10));
        // Brackets
        m.insert("(".to_string(), Value::Bracket("(".to_string()));
        m.insert(")".to_string(), Value::Bracket(")".to_string()));
        m.insert("{".to_string(), Value::Bracket("{".to_string()));
        m.insert("}".to_string(), Value::Bracket("}".to_string()));
        // Keywords
        m.insert("let".to_string(), Value::Symbol("let".to_string()));
        m.insert("if".to_string(), Value::Symbol("if".to_string()));
        m.insert("else".to_string(), Value::Symbol("else".to_string()));
        m.insert("true".to_string(), Value::Boolean(true));
        m.insert("false".to_string(), Value::Boolean(false));
        m.insert("for".to_string(), Value::Symbol("for".to_string()));
        m
    };
}

pub fn get_identifier(token: &str) -> Value {
    if let Some(symbol) = SYMBOLS.get(token) {
        symbol.clone()
    } else {
        Value::Identifier(token.to_string())
    }
}

pub fn get_operator(token: &str) -> Value {
    let ope_value = get_identifier(token);
    if let Value::Operator(..) = ope_value {
        ope_value
    } else {
        panic!("can't parse \"{}\" as operator", token);
    }
}

pub fn get_symbol(token: &str) -> Value {
    let symbol = get_identifier(token);
    if let Value::Symbol(..) = symbol {
        symbol
    } else {
        panic!("can't parse \"{}\" as symbol", token);
    }
}

pub fn get_bracket(token: &str) -> Value {
    let value = get_identifier(token);
    if let Value::Bracket(..) = value {
        value
    } else {
        panic!("can't parse \"{}\" as bracket", token);
    }
}

pub fn get_end_bracket(bra: &str) -> Option<String> {
    match bra {
        "(" => Some(")".to_string()),
        "{" => Some("}".to_string()),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub enum ValueType {
    Inference,
    Id(i64),
    None,
}

impl Element {
    fn print_with_prefix(&self, fmt: &mut fmt::Formatter, prefix: &str) -> fmt::Result {
        if let Err(err) = fmt.write_fmt(format_args!(
            "{}{:?} : {:?}",
            prefix, self.value, self.value_type
        )) {
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

impl std::string::ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Integer(value) => value.to_string(),
            Value::String(value) => value.to_string(),
            Value::Boolean(value) => value.to_string(),
            _ => String::new(),
        }
    }
}
