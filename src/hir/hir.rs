use std::{collections::HashMap, sync::{Arc, Mutex, RwLock, Weak}};

use super::Type;

#[derive(Debug, Clone)]
pub struct HirHub {
    pub hirs: Vec<Hir>,
    pub types: Vec<Type>,
}

impl HirHub {
    pub fn new() -> HirHub {
        HirHub{
            hirs: Vec::new(),
            types: Vec::new(),
        }
    }

    pub fn create_hir(&mut self, filename: &str, module_name: &str) -> &Hir {
        let hir = Hir::new(filename, module_name);
        self.hirs.push(hir);
        &self.hirs.last().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct Hir {  // file scope
    pub finename: String,
    pub module_name: String,
    pub dependency: Vec<Weak<Hir>>,
    pub module_symbols: HashMap<String,Definition>,
    pub body: Region,
}

impl Hir {
    pub fn new(finename: &str, module_name: &str) -> Hir {
        let mut body = Region::new();
        body.blocks.push(Block::new());
        Hir{
            finename: finename.to_string(),
            module_name: module_name.to_string(),
            dependency: Vec::new(),
            module_symbols: HashMap::new(),
            body,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: u64,
    pub public: bool,
    pub name: String,
    pub hash: (u64, u64, u64, u64),  // 256 bit
    pub expr: Weak<RwLock<Expression>>,
}

impl Definition {
    pub fn new(public: bool, name: &str, expr: Weak<RwLock<Expression>>) -> Definition {
        Definition{
            id: 0,
            public,
            name: name.to_string(),
            hash: (0, 0, 0, 0),
            expr,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression {
    pub body: ExpressionBody,
    pub ty: Type,
    pub location: Location,
}

impl Expression {
    pub fn new(body: ExpressionBody, ty: Type, location: Location) -> Expression {
        Expression{
            body,
            ty,
            location,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExpressionBody {
    None(),
    Variable(String),
    Identifier(String),
    Str{name: String, text: String},  // "string" ()
    I64(i64),
    Add(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // +
    Sub(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // -
    Mul(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // *
    Div(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // /
    // Parentheses(Arc<RwLock<Expression>>),  // ()
    Let(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // let l = r
    Assign(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // =
    Function {  // fn f(x: Type) -> Type { ... }
        name: String,
        args: Vec<Arc<RwLock<Expression>>>,
        body: Region,
        retern_type: Type,
    },
    BuiltinFunction(BuiltinFunction),  // fn f(x: Type) -> Type { ... }
    FunctionCall {  // fn f(x: Type) -> Type { ... }
        fn_def: Definition,
        args: Vec<Arc<RwLock<Expression>>>,
    },
}

impl ExpressionBody {
    pub fn to_expression(self) -> Expression {
        Expression::new(self, Type::Resolving(), Location::unknown())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Location {
    filename: String,
    line: u64,
    column: u64,
}

impl Location {
    pub fn unknown() -> Location {
        Location{
            filename: String::new(),
            line: 0,
            column: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Region {  // for code context
    pub blocks: Vec<Block>,
    pub symbols: HashMap<String,Definition>,
}

impl Region {
    pub fn new() -> Region {
        Region{
            blocks: Vec::new(),
            symbols: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub args: Vec<Arc<RwLock<Expression>>>,
    pub exprs: Vec<Arc<RwLock<Expression>>>,
    pub terminator: Option<Terminator>,
    pub location: Location,
}

impl Block {
    pub fn new() -> Block {
        Block{
            args: Vec::new(),
            exprs: Vec::new(),
            terminator: None,
            location: Location::unknown(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Terminator {
    Return,
    Goto,
    Branch,
}

#[derive(Debug, PartialEq, Clone)]
pub enum BuiltinFunction {
    Printf,
}

impl BuiltinFunction {
    pub fn to_expression(self) -> Expression {
        Expression::new(ExpressionBody::BuiltinFunction(self), Type::Resolving(), Location::unknown())
    }
}
