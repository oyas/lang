use std::{collections::HashMap, sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak}};

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
    pub module: Arc<RwLock<Module>>,
}

impl Hir {
    pub fn new(finename: &str, module_name: &str) -> Hir {
        let pre_body = Arc::new(RwLock::new(Region {
            blocks: Vec::new(),
            symbols: HashMap::new(),
            parent: None,
            module: None,
        }));
        pre_body.write().unwrap().blocks.push(Block::new(&pre_body));
        let body = Arc::new(RwLock::new(Region {
            blocks: Vec::new(),
            symbols: HashMap::new(),
            parent: None,
            module: None,
        }));
        body.write().unwrap().blocks.push(Block::new(&body));
        let module = Arc::new(RwLock::new(Module {
            finename: finename.to_string(),
            module_name: module_name.to_string(),
            dependency: Vec::new(),
            module_symbols: HashMap::new(),
            pre_body: Arc::clone(&pre_body),
            body,
        }));
        module.write().unwrap().pre_body.write().unwrap().module = Some(Arc::downgrade(&module));
        module.write().unwrap().body.write().unwrap().module = Some(Arc::downgrade(&module));
        module.write().unwrap().body.write().unwrap().parent = Some(Arc::downgrade(&pre_body));
        Hir { module }
    }

    pub fn read(&self) -> RwLockReadGuard<Module> {
        self.module.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<Module> {
        self.module.write().unwrap()
    }

    pub fn filename(&self) -> String {
        self.read().finename.clone()
    }

    pub fn module_name(&self) -> String {
        self.read().module_name.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Module {
    pub finename: String,
    pub module_name: String,
    pub dependency: Vec<Weak<Hir>>,
    pub module_symbols: HashMap<String,Definition>,
    pub pre_body: Arc<RwLock<Region>>,
    pub body: Arc<RwLock<Region>>,
}

#[derive(Debug, Clone)]
pub struct Definition {
    pub id: u64,
    pub public: bool,
    pub name: String,
    pub hash: (u64, u64, u64, u64),  // 256 bit
    pub expr: Weak<RwLock<Expression>>,
    pub region: Weak<RwLock<Region>>,
}

impl Definition {
    pub fn new(public: bool, name: &str, expr: &Arc<RwLock<Expression>>, region: &Arc<RwLock<Region>>) -> Definition {
        let expr = Arc::downgrade(expr);
        let region = Arc::downgrade(region);
        Definition{
            id: 0,
            public,
            name: name.to_string(),
            hash: (0, 0, 0, 0),
            expr,
            region,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expression {
    pub body: ExpressionBody,
    pub ty: Type,
    pub location: Location,
    pub region: Weak<RwLock<Region>>,
}

impl Expression {
    pub fn new(body: ExpressionBody, ty: Type, location: Location, region: &Arc<RwLock<Region>>) -> Expression {
        let region = Arc::downgrade(region);
        Expression{
            body,
            ty,
            location,
            region,
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
    Let(String, Arc<RwLock<Expression>>),  // let l = r
    Assign(Arc<RwLock<Expression>>, Arc<RwLock<Expression>>),  // =
    Function {  // fn f(x: Type) -> Type { ... }
        name: String,
        args: Vec<Arc<RwLock<Expression>>>,
        body: Arc<RwLock<Region>>,
        retern_type: Type,
    },
    BuiltinFunction(BuiltinFunction),  // fn f(x: Type) -> Type { ... }
    FunctionCall {  // fn f(x: Type) -> Type { ... }
        fn_def: Definition,
        args: Vec<Arc<RwLock<Expression>>>,
    },
}

impl ExpressionBody {
    pub fn to_expression(self, region: &Arc<RwLock<Region>>) -> Arc<RwLock<Expression>> {
        Arc::new(RwLock::new(Expression::new(
            self,
            Type::Resolving(),
            Location::unknown(),
            region
        )))
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
    pub parent: Option<Weak<RwLock<Region>>>,
    pub module: Option<Weak<RwLock<Module>>>,
}

impl Region {
    pub fn new(parent: &Arc<RwLock<Region>>) -> Region {
        let module = parent.read().unwrap().module.clone();
        Region{
            blocks: Vec::new(),
            symbols: HashMap::new(),
            parent: Some(Arc::downgrade(parent)),
            module,
        }
    }

    pub fn find_symbol(&self, name: &str) -> Option<Definition> {
        if let Some(def) = self.symbols.get(name) {
            return Some(def.clone());
        }
        if let Some(parent) = &self.parent {
            return parent.upgrade().unwrap().read().unwrap().find_symbol(name);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub args: Vec<Arc<RwLock<Expression>>>,
    pub exprs: Vec<Arc<RwLock<Expression>>>,
    pub terminator: Option<Terminator>,
    pub location: Location,
    pub region: Weak<RwLock<Region>>,
}

impl Block {
    pub fn new(region: &Arc<RwLock<Region>>) -> Block {
        let region = Arc::downgrade(region);
        Block{
            args: Vec::new(),
            exprs: Vec::new(),
            terminator: None,
            location: Location::unknown(),
            region,
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
    pub fn to_expression(self, region: &Arc<RwLock<Region>>) -> Arc<RwLock<Expression>> {
        Arc::new(RwLock::new(Expression::new(
            ExpressionBody::BuiltinFunction(self),
            Type::Resolving(),
            Location::unknown(),
            region
        )))
    }
}
