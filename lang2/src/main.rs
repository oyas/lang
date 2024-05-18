use std::io::Write;

mod ast;
mod backend;
mod parser;

fn main() {
    println!("Hello, world!");
    let ret = backend::llvm::compile();
    println!("{:?}", ret);
    repl();
}

fn repl() {
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).ok();
        if s == "" {
            println!("quit");
            break;
        }
        let ret = match parser::parse(&s) {
            Ok((_, r)) => r,
            Err(e) => {
                println!("parse error! {:?}", e);
                continue;
            }
        };
        println!("parsed: {:?}", ret)
    }
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn test() {
        assert_eq!(4, 4);
    }
}