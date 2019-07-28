use std::collections::HashMap;

use super::element;

#[derive(Debug, Clone)]
pub struct Scope {
    values: HashMap<String, element::Element>,
}

impl Scope {
    pub fn new() -> Scope {
        Scope{ values: HashMap::new() }
    }

    pub fn get(&self, key: &str) -> &element::Element {
        if let Some(el) = self.values.get(key) {
            el
        } else {
            panic!("Access to undefined symbol \"{}\"", key);
        }
    }

    pub fn set(&mut self, key: &str, value: element::Element) {
        self.values.insert(key.to_string(), value);
    }
}

pub fn eval(el: &element::Element, scope: &mut Scope) -> Option<element::Element> {
    // for '+', '-', '*', '/'
    let calc = |el: &element::Element, scope: &mut Scope, func: fn(l: i64, r: i64) -> i64| -> Option<element::Element> {
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

    // for 'let'
    let ope_let = |el: &element::Element, scope: &mut Scope, update: bool| -> Option<element::Element> {
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
        element::Value::Integer(_) => {
            Some(el.clone())
        }
        x if *x == element::get_operator("+") => {
            calc(el, scope, |l, r| l + r)
        }
        x if *x == element::get_operator("-") => {
            calc(el, scope, |l, r| l - r)
        }
        x if *x == element::get_operator("*") => {
            calc(el, scope, |l, r| l * r)
        }
        x if *x == element::get_operator("/") => {
            calc(el, scope, |l, r| {
                if r == 0 {
                    panic!("divide by zero");
                }
                l / r
            })
        }
        x if *x == element::get_operator("=") => {
            ope_let(el, scope, true)
        }
        x if *x == element::get_symbol("let") => {
            ope_let(el, scope, false)
        }
        element::Value::Identifier(id) => {
            Some(scope.get(id).clone())
        }
        x if *x == element::get_bracket("(") => {
            eval(el.childlen.first().unwrap(), scope)
        }
        _ => {
            panic!("Invalid syntax");
        }
    }
}
