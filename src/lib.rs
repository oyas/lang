#![allow(dead_code)]

use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Stdin;

mod parser;
mod evaluator;

fn read_tokens(stream: &mut BufRead) -> Vec<String> {
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

pub fn run(file_name: &str, show_log: bool) -> Option<parser::element::Element> {
    // read from source
    let tokens: Vec<String> = if file_name.is_empty() {
        let stdin: Stdin = std::io::stdin();
        let mut buf = BufReader::new(stdin);
        read_tokens(&mut buf)
    } else {
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

    if show_log {
        println!("{:?}", tokens);
    }

    // parse
    let el = parser::parse_element(&tokens, &mut 0, "", -1, "", false);

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
