#![allow(dead_code)]

use std::io::BufReader;
use std::io::BufRead;
use std::io::Stdin;
use std::io;
use std::fs::File;
use std::env;

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

fn parse_element(tokens: &[String], pos: &mut usize, limit: i32, mut end_bracket: String) -> Option<element::Element> {
    if *pos >= tokens.len() {
        return None;
        //panic!("out of bounds");
    }

    let mut result: element::Element = if limit == -1 {
        // single file scope
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
                //println!("bracket childlen = {:?}", next);
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
        _ => {}
    }

    // check if the next token is operator
    loop {
        if *pos >= tokens.len() {
            break;
        } else if let Some(next_element) = element::get_element(&tokens[*pos]) {
            if let element::Value::Operator(..) = next_element.value {
                if let Some(next) = parse_element(tokens, pos, limit, end_bracket.clone()) {
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

//fn reorder_elelemnt<'a>(tokens: &[String], pos: &mut usize, el: &'a mut element::Element, el_ope: &'a mut element::Element) -> &'a mut element::Element {
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

fn eval(el: &element::Element) -> i64 {
    match &el.value {
        element::Value::FileScope() => {
            let mut ret = 0;
            for el in &el.childlen {
                ret = eval(el);
            }
            ret
        }
        element::Value::Integer(n) => {
            *n
        }
        x if *x == element::OPERATORS["+"] => {
            if let [l, r] = &el.childlen[..] {
                eval(l) + eval(r)
            } else {
                panic!("Invalid syntax");
            }
        }
        x if *x == element::OPERATORS["-"] => {
            if let [l, r] = &el.childlen[..] {
                eval(l) - eval(r)
            } else {
                panic!("Invalid syntax");
            }
        }
        x if *x == element::OPERATORS["*"] => {
            if let [l, r] = &el.childlen[..] {
                eval(l) * eval(r)
            } else {
                panic!("Invalid syntax");
            }
        }
        x if *x == element::OPERATORS["/"] => {
            if let [l, r] = &el.childlen[..] {
                let r_val = eval(r);
                if r_val == 0 {
                    panic!("divide by zero");
                }
                eval(l) / r_val
            } else {
                panic!("Invalid syntax");
            }
        }
        element::Value::Brackets(x) if x == "(" => {
            eval(el.childlen.first().unwrap())
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
    let mut pos: usize = 0;
    let el = parse_element(&tokens, &mut pos, -1, String::new());
    //println!("parse_element: {:#?}", el);

    // evaluate
    if let Some(el) = el {
        println!("parse_element:\n{}", el);
        let result = eval(&el);
        println!("eval: {:#?}", result);
    }
}
