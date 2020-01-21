use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::parser::element;

#[derive(Debug, Clone)]
pub struct EvaledElement {
    pub el: element::Element,
    scope: Option<Scope>,
}

impl EvaledElement {
    pub fn new(el: element::Element) -> EvaledElement {
        EvaledElement {
            el: el,
            scope: None,
        }
    }

    pub fn from_value(value: element::Value) -> EvaledElement {
        EvaledElement {
            el: element::Element::new(value),
            scope: None,
        }
    }
}

#[derive(Debug)]
struct ScopeInner {
    values: HashMap<String, Rc<RefCell<EvaledElement>>>,
    parent: Option<Scope>,
}

#[derive(Debug, Clone)]
pub struct Scope(Rc<RefCell<ScopeInner>>);

impl Scope {
    pub fn new() -> Scope {
        init_scope()
    }

    pub fn empty() -> Scope {
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

    pub fn get(&self, key: &str) -> EvaledElement {
        match self.get_mut(key) {
            Some(el) => el.borrow_mut().clone(),
            None => panic!("Access to undefined symbol \"{}\"", key),
        }
    }

    pub fn set(&self, key: &str, value: EvaledElement) {
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

    pub fn get_mut(&self, key: &str) -> Option<Rc<RefCell<EvaledElement>>> {
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

pub fn init_scope() -> Scope {
    let insert_inner_function = |func_map: &mut HashMap<String, Rc<RefCell<EvaledElement>>>,
                                 name: &str| {
        let val_el = element::Element::new(element::get_identifier("value"));
        let mut bra_el = element::Element::new(element::get_bracket("("));
        bra_el.childlen.push(val_el);
        let body_el = element::Element::new(element::Value::InnerFunction(name.to_string()));
        let mut fun_el = element::Element::new(element::get_symbol("fun"));
        fun_el.childlen.push(bra_el);
        fun_el.childlen.push(body_el);
        func_map.insert(
            name.to_string(),
            Rc::new(RefCell::new(EvaledElement {
                el: fun_el,
                scope: Some(Scope::empty()),
            })),
        )
    };

    let mut functions: HashMap<String, Rc<RefCell<EvaledElement>>> = HashMap::new();
    insert_inner_function(&mut functions, "print");

    let inner = ScopeInner {
        values: functions,
        parent: None,
    };
    Scope(Rc::new(RefCell::new(inner)))
}

fn inner_function(name: &str, _el: &EvaledElement, scope: &mut Scope) -> Option<EvaledElement> {
    match name {
        "print" => {
            let a = scope.get("value");
            print!("{}", a.el.value.to_string());
            Some(EvaledElement::from_value(element::Value::None))
        }
        _ => panic!("Internal error."),
    }
}

pub fn eval(el: &element::Element, scope: &mut Scope) -> Option<EvaledElement> {
    eval_inner(&EvaledElement::new(el.clone()), scope)
}

pub fn eval_inner(element: &EvaledElement, scope: &mut Scope) -> Option<EvaledElement> {
    let el = &element.el;

    // for '+', '-', '*', '/'
    let calc = |el: &element::Element,
                scope: &mut Scope,
                func: fn(l: i64, r: i64) -> i64|
     -> Option<EvaledElement> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let (Some(mut l), Some(r)) = (eval(el_l, scope), eval(el_r, scope)) {
                if let (element::Value::Integer(int_l), element::Value::Integer(int_r)) =
                    (l.el.value.clone(), r.el.value)
                {
                    let result = func(int_l, int_r);
                    l.el.value = element::Value::Integer(result);
                }
                Some(l)
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
     -> Option<EvaledElement> {
        if let [el_l, el_r] = &el.childlen[..] {
            if let (Some(l), Some(r)) = (eval(el_l, scope), eval(el_r, scope)) {
                let value = func(l.el.value, r.el.value);
                Some(EvaledElement::from_value(element::Value::Boolean(value)))
            } else {
                None
            }
        } else {
            panic!("Invalid syntax");
        }
    };

    // for 'let'
    let ope_let =
        |el: &element::Element, scope: &mut Scope, update: bool| -> Option<EvaledElement> {
            if let [el_l, el_r] = &el.childlen[..] {
                if let (element::Value::Identifier(id_l), Some(r)) =
                    (el_l.value.clone(), eval(el_r, scope))
                {
                    if update {
                        scope.get(&id_l);
                    }
                    scope.set(&id_l, r.clone());
                    Some(r)
                } else {
                    None
                }
            } else {
                panic!("Invalid syntax");
            }
        };

    match &el.value {
        element::Value::EvalScope() => {
            let mut ret = None;
            for el in &el.childlen {
                if let element::Value::Import {..} = el.value {
                    ret = eval(&el.childlen.first().unwrap(), scope);
                }
            }
            for el in &el.childlen {
                if let element::Value::FileScope() = el.value {
                    ret = eval(el, scope);
                }
            }
            ret
        }
        element::Value::FileScope() => {
            let mut ret = None;
            for el in &el.childlen {
                ret = eval(el, scope);
            }
            ret
        }
        element::Value::Import {..} => { None }
        element::Value::Integer(_) | element::Value::Boolean(_) | element::Value::String(_) => {
            Some(EvaledElement::new(el.clone()))
        }
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
                Some(c) => match c.el.value {
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
                    Some(c) => match c.el.value {
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
        x if *x == element::get_symbol("fun") => {
            if el.childlen.len() < 2 {
                panic!("Invalid syntax")
            }
            let mut new_el = EvaledElement::new(el.clone());
            new_el.scope = Some(scope.clone());
            Some(new_el)
        }
        element::Value::FunctionCall(id) => {
            let fun = scope.get(id);
            if fun.el.value != element::get_symbol("fun") {
                panic!("Invalid syntax: non-existent function call")
            }
            let params = &fun.el.childlen.first().unwrap().childlen;
            let body = fun.el.childlen.last().unwrap();
            if el.childlen.len() != params.len() {
                panic!(
                    "Invalid syntax: num of parameter is not mutch {} != {}",
                    el.childlen.len(),
                    params.len()
                )
            }
            let mut eval_params: HashMap<String, EvaledElement> = HashMap::new();
            for (i, param) in params.iter().enumerate() {
                match eval(el.childlen.get(i).unwrap(), scope) {
                    Some(value) => match &param.value {
                        element::Value::Identifier(id) => {
                            eval_params.insert(id.clone(), value.clone());
                        }
                        _ => panic!("Invalid syntax"),
                    },
                    None => panic!("Invalid syntax"),
                }
            }
            fun.scope.unwrap().new_scope(|mut fun_scope| {
                for (id, value) in &eval_params {
                    fun_scope.set(&id, value.clone())
                }
                eval(body, &mut fun_scope)
            })
        }
        element::Value::InnerFunction(ref name) => inner_function(name, element, scope),
        element::Value::Identifier(id) => Some(scope.get(id)),
        x if *x == element::get_bracket("(") => eval(el.childlen.first().unwrap(), scope),
        x if *x == element::get_bracket("{") => {
            let mut res = None;
            for el in &el.childlen {
                res = eval(el, scope)
            }
            res
        }
        _ => {
            panic!("Invalid syntax: eval {}", el);
        }
    }
}
