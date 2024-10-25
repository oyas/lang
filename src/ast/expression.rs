#![allow(unused)]

#[derive(Debug, PartialEq)]
pub enum Expression {
    Resolving(),  // used for left term before reordering
    None(),
    Identifier(String),
    I64(i64),
    Add(Box<Expression>, Box<Expression>),  // +
    Sub(Box<Expression>, Box<Expression>),  // -
    Mul(Box<Expression>, Box<Expression>),  // *
    Parentheses(Box<Expression>),  // ()
}

impl Expression {
    fn priority(&self) -> i32 {
        match self {
            Self::Resolving() => 0,
            Self::None() => 0,
            Self::Identifier(_) => 0,
            Self::I64(_) => 0,
            Self::Add(_, _) => -50,
            Self::Sub(_, _) => -50,
            Self::Mul(_, _) => -20,
            Self::Parentheses(_) => 0,
        }
    }

    fn children(&self) -> Vec<&Box<Expression>> {
        match self {
            Self::Add(a, b) => vec![a, b],
            Self::Sub(a, b) => vec![a, b],
            Self::Mul(a, b) => vec![a, b],
            Self::Parentheses(a) => vec![a],
            _ => vec![],
        }
    }

    fn replace_first_child(self, l: Expression) -> Self {
        let l = Box::new(l);
        match self {
            Self::Add(_a, b) => Self::Add(l, b),
            Self::Sub(_a, b) => Self::Sub(l, b),
            Self::Mul(_a, b) => Self::Mul(l, b),
            Self::Parentheses(_a) => Self::Parentheses(l),
            _ => panic!("This does not have children."),
        }
    }

    pub fn reorder(self, r: Expression) -> Self {
        if self.priority() >= r.priority() {
            return r.replace_first_child(self)
        }
        match self {
            Self::Add(a, b) => Self::Add(a, Box::new(b.reorder(r))),
            Self::Sub(a, b) => Self::Sub(a, Box::new(b.reorder(r))),
            Self::Mul(a, b) => Self::Mul(a, Box::new(b.reorder(r))),
            _ => panic!("This is not operator."),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_first_child() {
        let a = Expression::Add(
            Box::new(Expression::None()),
            Box::new(Expression::None()),
        );
        println!("{:?}", a);
        let b = a.replace_first_child(Expression::I64(1));
        println!("{:?}", b);
        assert_eq!(b, Expression::Add(
            Box::new(Expression::I64(1)),
            Box::new(Expression::None()),
        ))
    }

    #[test]
    fn reorder() {
        let a = Expression::Add(
            Box::new(Expression::None()),
            Box::new(Expression::None()),
        );
        let b = Expression::Mul(
            Box::new(Expression::Resolving()),
            Box::new(Expression::None()),
        );
        println!("{:?} {:?}", a, b);
        let c = a.reorder(b);
        println!("{:?}", c);
        assert_eq!(c, Expression::Add(
            Box::new(Expression::None()),
            Box::new(Expression::Mul(
                Box::new(Expression::None()),
                Box::new(Expression::None()),
            )),
        ))
    }

    #[test]
    fn reorder2() {
        let a = Expression::Add(
            Box::new(Expression::I64(1)),
            Box::new(Expression::I64(2)),
        );
        let b = Expression::Add(
            Box::new(Expression::Resolving()),
            Box::new(Expression::I64(3)),
        );
        println!("{:?} {:?}", a, b);
        let c = a.reorder(b);
        println!("{:?}", c);
        assert_eq!(c, Expression::Add(
            Box::new(Expression::Add(
                Box::new(Expression::I64(1)),
                Box::new(Expression::I64(2)),
            )),
            Box::new(Expression::I64(3)),
        ))
    }
}