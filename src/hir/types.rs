
#[derive(Debug, PartialEq)]
pub enum Type {
    Resolving(),
    BasicType(BasicType),
    Array(Box<Type>),
    Struct(Vec<(String, u64)>),
    Id(i64),
}

#[derive(Debug, PartialEq)]
pub enum BasicType {
    I64,
    Bool,
    Void,
}
