#![allow(dead_code)]

use std::io::BufReader;
use std::io::BufRead;
use std::io::Stdin;
use std::io;
use std::fs::File;
use std::env;
use std::collections::HashMap;

mod parser;
use parser::element;

fn read_tokens(stream: &mut BufRead) -> Vec<String> {
    println!("reading");
    let mut tokens: Vec<String> = vec![];
    let mut buffer = String::new();
    loop {
        match stream.read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let mut token = String::new();
                buffer.push('\n');
                for c in buffer.chars() {
                    if c.is_whitespace() {
                        if !token.is_empty() {
                            tokens.push(token.clone());
                            token.clear();
                        }
                    } else if c.is_ascii_punctuation() {
                        if !token.is_empty() {
                            tokens.push(token.clone());
                            token.clear();
                        }
                        tokens.push(c.to_string());
                    } else {
                        token.push(c);
                    }
                }
                tokens.push("\n".to_string());
                println!("{:?}", buffer);
                buffer.clear();
            }
            Err(e) => {
                println!("Error occurred while read_tokens: {}", e);
                break;
            }
        }
    }
    return tokens;
}

#[derive(Debug, Clone)]
pub struct Scope {
    values: HashMap<String, element::Element>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope{ values: HashMap::new() }
    }

    pub fn get(&self, key: &str) -> &element::Element {
        if let Some(el) = self.values.get(key) {
            el
        } else {
            panic!("Access to undefined symbol \"{}\"", key);
        }
    }

    pub fn set(&mut self, key: &str, value: element::Element) {
        self.values.insert(key.to_string(), value);
    }
}

fn parse_element(tokens: &[String], pos: &mut usize, limit: i32, mut end_bracket: String) -> Option<element::Element> {
    if *pos >= tokens.len() {
        return None;
        //panic!("out of bounds");
    }

    let mut result: element::Element = if limit == -1 {
        // file level scope
        element::Element{
            value: element::Value::FileScope(),
            value_type: element::ValueType::None,
            childlen: Vec::new(),
        }
    } else {
        // read current token
        let token = &tokens[*pos];
        *pos += 1;
        if limit > 0 && token == "\n" {
            return parse_element(tokens, pos, limit, end_bracket);
        }
        // parse
        if let Some(el) = element::get_element(token) {
            el
        } else {
            return None;
        }
    };

    if let element::Value::Operator(_, priority) = result.value {
        // check priority of operator
        if priority < limit {
            *pos -= 1;
            return None;
        }
    } else if let element::Value::Brackets(ref bra) = result.value {
        // check end of bracket
        if *bra == end_bracket {
            return Some(result);
        } else {
            if let Some(eb) = element::get_end_bracket(bra) {
                end_bracket = eb;
            } else {
                panic!("Invalid syntax '{}' {}", bra, end_bracket);
            }
        }
    }

    // read childlen elements
    match result.value {
        element::Value::Operator(..) | element::Value::EndLine() => {
            return Some(result);
        }
        element::Value::Brackets(..) => {
            while let Some(next) = parse_element(tokens, pos, 0, end_bracket.clone()) {
                match next.value {
                    element::Value::Brackets(ref bra) => {
                        if *bra == end_bracket {
                            break;
                        } else {
                            result.childlen.push(next);
                        }
                    }
                    element::Value::EndLine() => {}
                    _ => {
                        result.childlen.push(next);
                    }
                }
            }
        }
        element::Value::FileScope() => {
            while let Some(next) = parse_element(tokens, pos, 0, end_bracket.clone()) {
                match next.value {
                    element::Value::EndLine() => {}
                    _ => {
                        result.childlen.push(next);
                    }
                }
            }
        }
        ref x if *x == element::get_symbol("let") => {
            if let Some(next) = parse_element(tokens, pos, 1, String::new()) {
                match next.value {
                    ref ope if *ope == element::get_operator("=") => {
                        result.childlen = next.childlen;
                    }
                    _ => {
                        panic!("Invalid syntax: operator 'let' can't found '=' token.");
                    }
                }
            }
        }
        _ => {}
    }

    // check if the next token is operator
    loop {
        if *pos >= tokens.len() {
            break;
        } else if let Some(next_element) = element::get_element(&tokens[*pos]) {
            if let element::Value::Operator(_, priority) = next_element.value {
                if priority < limit {  // check priority of operator
                    break;
                } else if let Some(next) = parse_element(tokens, pos, limit, end_bracket.clone()) {
                    result = reorder_elelemnt(tokens, pos, result, next, end_bracket.clone());
                } else {
                    panic!("Invalid syntax");
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return Some(result);
}

// use to parse operator
fn reorder_elelemnt(tokens: &[String], pos: &mut usize, mut el: element::Element, mut el_ope: element::Element, end_bracket: String) -> element::Element {
    if let element::Value::Operator(_, priority_ope) = el_ope.value {
        // finding left element
        if let element::Value::Operator(_, priority) = el.value {
            if priority_ope > priority {
                // el_ope is child node
                if let Some(el_right) = el.childlen.pop() {
                    let res = reorder_elelemnt(tokens, pos, el_right, el_ope, end_bracket);
                    el.childlen.push(res);
                    return el;
                } else {
                    panic!("Invalid syntax");
                }
            }
        }

        // el_ope is parent node. el is left node
        el_ope.childlen.push(el);
        // read right token
        let next = parse_element(tokens, pos, priority_ope, end_bracket);
        if let Some(next) = next {
            el_ope.childlen.push(next);
        } else {
            println!("Invalid syntax.");
        }
        return el_ope;
    } else {
        panic!("Invalid syntax");
    }
}

fn eval(el: &element::Element, scope: &mut Scope) -> Option<element::Element> {
    // for '+', '-', '*', '/'
    let calc = |el: &element::Element, scope: &mut Scope, func: fn(l: i64, r: i64) -> i64| -> Option<element::Element> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let Some(mut l) = eval(el_l, scope) {
                if let Some(r) = eval(el_r, scope) {
                    if let element::Value::Integer(int_l) = l.value {
                        if let element::Value::Integer(int_r) = r.value {
                            l.value = element::Value::Integer(func(int_l, int_r));
                        }
                    }
                    Some(l)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            panic!("Invalid syntax");
        }
    };

    // for 'let'
    let ope_let = |el: &element::Element, scope: &mut Scope, update: bool| -> Option<element::Element> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let Some(r) = eval(el_r, scope) {
                if let element::Value::Identifier(id_l) = &el_l.value {
                    if update {
                        scope.get(id_l);
                    }
                    scope.set(&id_l, r.clone());
                    return Some(r);
                }
            }
            None
        } else {
            panic!("Invalid syntax");
        }
    };

    match &el.value {
        element::Value::FileScope() => {
            let mut ret = None;
            for el in &el.childlen {
                ret = eval(el, scope);
            }
            ret
        }
        element::Value::Integer(_) => {
            Some(el.clone())
        }
        x if *x == element::get_operator("+") => {
            calc(el, scope, |l, r| l + r)
        }
        x if *x == element::get_operator("-") => {
            calc(el, scope, |l, r| l - r)
        }
        x if *x == element::get_operator("*") => {
            calc(el, scope, |l, r| l * r)
        }
        x if *x == element::get_operator("/") => {
            calc(el, scope, |l, r| {
                if r == 0 {
                    panic!("divide by zero");
                }
                l / r
            })
        }
        x if *x == element::get_operator("=") => {
            ope_let(el, scope, true)
        }
        x if *x == element::get_symbol("let") => {
            ope_let(el, scope, false)
        }
        element::Value::Identifier(id) => {
            Some(scope.get(id).clone())
        }
        element::Value::Brackets(x) if x == "(" => {
            eval(el.childlen.first().unwrap(), scope)
        }
        _ => {
            panic!("Invalid syntax");
        }
    }
}


fn main() {
    let args: Vec<String> = env::args().collect::<Vec<String>>();

    let tokens: Vec<String> = if args.len() <= 1 {
        let stdin: Stdin = std::io::stdin();
        let mut buf = BufReader::new(stdin);
        read_tokens(&mut buf)
    } else {
        let file_name = &args[1];
        match File::open(file_name) {
            Ok(file) => {
                let mut buf = BufReader::new(file);
                read_tokens(&mut buf)
            }
            Err(e) => {
                println!("Error occurred while reading {}: {}", file_name, e);
                let mut buf = BufReader::new(io::empty());
                read_tokens(&mut buf)
            }
        }
    };

    for input in &tokens {
        println!("{:#?}", input);
    }

    // parse
    let el = parse_element(&tokens, &mut 0, -1, String::new());

    // evaluate
    if let Some(el) = el {
        println!("parse_element:\n{}", el);
        let result = eval(&el, &mut Scope::new());
        println!("eval: {:?}", result.unwrap().value);
    }
}
