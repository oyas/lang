#![allow(dead_code)]

use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Stdin;
use std::path::Path;

mod evaluator;
mod parser;

fn read_tokens(stream: &mut dyn BufRead) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];
    let mut buffer = String::new();
    loop {
        match stream.read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                tokens.append(&mut parser::line_to_tokens(&buffer));
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

fn read_rawfile(file_name: &str) -> Vec<String> {
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
}

pub fn read(
    tokens: &[String],
    current_dir: &Path,
    ignore_imports: &HashSet<String>,
) -> Option<parser::element::Element> {
    let mut ignore_imports = ignore_imports.clone();
    let mut eval_el = parser::element::Element::new(parser::element::Value::EvalScope());
    let el = parser::parse_element(tokens, &mut 0, "", -1, "", false);
    if let Some(el) = &el {
        for cl in &el.childlen {
            if let parser::element::Value::Import { path } = &cl.value {
                if ignore_imports.contains(path) {
                    continue;
                }
                ignore_imports.insert(path.clone());
                let file_name = current_dir.join(path);
                let tokens = read_rawfile(&file_name.to_str().unwrap());
                if let Some(file_el) = read(&tokens, current_dir, &ignore_imports) {
                    for child_el in file_el.childlen {
                        match &child_el.value {
                            parser::element::Value::Import { path } => {
                                ignore_imports.insert(path.clone());
                                eval_el.childlen.push(child_el);
                            }
                            parser::element::Value::FileScope() => {
                                let mut new_cl = cl.clone();
                                new_cl.childlen.push(child_el);
                                eval_el.childlen.push(new_cl);
                            }
                            _ => {
                                panic!("Internal Error");
                            }
                        }
                    }
                } else {
                    panic!("cannot import {}", file_name.to_str().unwrap());
                }
            }
        }
        eval_el.childlen.push(el.clone());
        Some(eval_el)
    } else {
        None
    }
}

pub fn run(file_name: &str, show_log: bool) -> Option<parser::element::Element> {
    // read from source
    let tokens: Vec<String> = if file_name.is_empty() {
        let stdin: Stdin = std::io::stdin();
        let mut buf = BufReader::new(stdin);
        read_tokens(&mut buf)
    } else {
        read_rawfile(file_name)
    };

    if show_log {
        println!("{:?}", tokens);
    }

    let current_dir = if file_name.is_empty() {
        Path::new(".")
    } else {
        Path::new(file_name).parent().unwrap()
    };

    // parse
    let el = read(&tokens, &current_dir, &HashSet::new());

    // evaluate
    if let Some(el) = &el {
        if show_log {
            println!("parse_element:\n{}", el);
        }
        let result = evaluator::eval(&el, &mut evaluator::Scope::new());
        if show_log {
            println!("result: {:?}", result);
        }
        result.and_then(|res| Some(res.el))
    } else {
        None
    }
}
