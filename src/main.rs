#![allow(dead_code)]

use std::io::BufReader;
use std::io::BufRead;
use std::io::Stdin;
use std::io;
use std::fs::File;
use std::env;

mod parser;

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

    // parse
    let el = parser::parse_element(&tokens, &mut 0, -1, String::new());

    // evaluate
    if let Some(el) = el {
        println!("parse_element:\n{}", el);
        let result = parser::eval(&el, &mut parser::Scope::new());
        println!("eval: {:?}", result.unwrap().value);
    }
}
