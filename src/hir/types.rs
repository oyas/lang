
#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Resolving(),
    BasicType(BasicType),
    Array(Box<Type>),
    Struct(Vec<(String, u64)>),
    Function(Vec<Type>, Vec<Type>),  // args, return
    Id(i64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum BasicType {
    Void,
    I64,
    Bool,
    Str,
}
