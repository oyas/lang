use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::element;

#[derive(Debug)]
struct ScopeInner {
    values: HashMap<String, Rc<RefCell<element::Element>>>,
    parent: Option<Scope>,
}

#[derive(Debug)]
pub struct Scope(Rc<RefCell<ScopeInner>>);

impl Scope {
    pub fn new() -> Scope {
        Scope(Rc::new(RefCell::new(ScopeInner {
            values: HashMap::new(),
            parent: None,
        })))
    }

    pub fn new_scope<T>(&self, func: impl Fn(Scope) -> T) -> T {
        let scope = Scope(Rc::new(RefCell::new(ScopeInner {
            values: HashMap::new(),
            parent: Some(Scope(Rc::clone(&self.0))),
        })));
        func(scope)
    }

    pub fn get(&self, key: &str) -> element::Element {
        match self.get_mut(key) {
            Some(el) => el.borrow_mut().clone(),
            None => panic!("Access to undefined symbol \"{}\"", key),
        }
    }

    pub fn set(&self, key: &str, value: element::Element) {
        match self.get_mut(key) {
            Some(el) => *el.borrow_mut() = value,
            None => {
                self.0
                    .borrow_mut()
                    .values
                    .insert(key.to_string(), Rc::new(RefCell::new(value)));
            }
        }
    }

    pub fn get_mut(&self, key: &str) -> Option<Rc<RefCell<element::Element>>> {
        let mut scope = self.0.borrow_mut();
        match &scope.values.get(key) {
            Some(el) => Some(Rc::clone(el)),
            None => match &scope.parent {
                Some(parent) => match parent.get_mut(key) {
                    Some(el) => {
                        scope.values.insert(key.to_string(), Rc::clone(&el));
                        Some(el)
                    }
                    None => None,
                },
                None => None,
            },
        }
    }
}

pub fn eval(el: &element::Element, scope: &mut Scope) -> Option<element::Element> {
    // for '+', '-', '*', '/'
    let calc = |el: &element::Element,
                scope: &mut Scope,
                func: fn(l: i64, r: i64) -> i64|
     -> Option<element::Element> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let Some(mut l) = eval(el_l, scope) {
                if let Some(r) = eval(el_r, scope) {
                    if let element::Value::Integer(int_l) = l.value {
                        if let element::Value::Integer(int_r) = r.value {
                            l.value = element::Value::Integer(func(int_l, int_r));
                        }
                    }
                    Some(l)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            panic!("Invalid syntax");
        }
    };

    // for '==', '!='
    let comp = |el: &element::Element,
                scope: &mut Scope,
                func: fn(l: element::Value, r: element::Value) -> bool|
     -> Option<element::Element> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let Some(l) = eval(el_l, scope) {
                if let Some(r) = eval(el_r, scope) {
                    let value = func(l.value, r.value);
                    Some(element::Element {
                        value: element::Value::Boolean(value),
                        value_type: element::ValueType::None,
                        childlen: Vec::new(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            panic!("Invalid syntax");
        }
    };

    // for 'let'
    let ope_let =
        |el: &element::Element, scope: &mut Scope, update: bool| -> Option<element::Element> {
            if let [el_l, el_r] = &el.childlen[..] {
                if let Some(r) = eval(el_r, scope) {
                    if let element::Value::Identifier(id_l) = &el_l.value {
                        if update {
                            scope.get(id_l);
                        }
                        scope.set(&id_l, r.clone());
                        return Some(r);
                    }
                }
                None
            } else {
                panic!("Invalid syntax");
            }
        };

    match &el.value {
        element::Value::FileScope() => {
            let mut ret = None;
            for el in &el.childlen {
                ret = eval(el, scope);
            }
            ret
        }
        element::Value::Integer(_) | element::Value::Boolean(_) => Some(el.clone()),
        x if *x == element::get_operator("+") => calc(el, scope, |l, r| l + r),
        x if *x == element::get_operator("-") => calc(el, scope, |l, r| l - r),
        x if *x == element::get_operator("*") => calc(el, scope, |l, r| l * r),
        x if *x == element::get_operator("/") => calc(el, scope, |l, r| {
            if r == 0 {
                panic!("divide by zero");
            }
            l / r
        }),
        x if *x == element::get_operator("==") => comp(el, scope, |l, r| l == r),
        x if *x == element::get_operator("!=") => comp(el, scope, |l, r| l != r),
        x if *x == element::get_operator("=") => ope_let(el, scope, true),
        x if *x == element::get_symbol("let") => ope_let(el, scope, false),
        x if *x == element::get_symbol("if") => match &el.childlen.first() {
            Some(condition) => scope.new_scope(|mut scope| match eval(condition, &mut scope) {
                Some(c) => match c.value {
                    element::Value::Boolean(true) => match el.childlen.get(1) {
                        Some(el) => eval(el, &mut scope),
                        None => panic!("Invalid syntax"),
                    },
                    element::Value::Boolean(false) => match el.childlen.get(2) {
                        Some(el) => eval(el, &mut scope),
                        None => panic!("Invalid syntax"),
                    },
                    _ => panic!("Invalid syntax"),
                },
                None => panic!("Invalid syntax"),
            }),
            None => panic!("Invalid syntax"),
        },
        x if *x == element::get_symbol("for") => match &el.childlen.first() {
            Some(condition) => scope.new_scope(|mut scope| {
                while match eval(condition, &mut scope) {
                    Some(c) => match c.value {
                        element::Value::Boolean(c) => c,
                        _ => panic!("Invalid syntax"),
                    },
                    None => panic!("Invalid syntax"),
                } {
                    match el.childlen.get(1) {
                        Some(el_scope) => eval(el_scope, &mut scope),
                        None => panic!("Invalid syntax"),
                    };
                }
                None
            }),
            None => panic!("Invalid syntax"),
        },
        element::Value::Identifier(id) => Some(scope.get(id).clone()),
        x if *x == element::get_bracket("(") => eval(el.childlen.first().unwrap(), scope),
        x if *x == element::get_bracket("{") => {
            let mut res = None;
            for el in &el.childlen {
                res = eval(el, scope)
            }
            res
        }
        _ => {
            panic!("Invalid syntax");
        }
    }
}
