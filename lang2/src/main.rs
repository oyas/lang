mod ast;
mod backend;
mod parser;
mod compiler;

fn main() {
    // println!("Hello, world!");
    // let ret = backend::llvm::compile();
    // println!("{:?}", ret);
    compiler::repl();
}

#[cfg(test)]
pub mod tests {
    #[test]
    pub fn test() {
        assert_eq!(4, 4);
    }
}