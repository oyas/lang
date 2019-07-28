use super::element;

pub fn line_to_tokens(buffer: &str) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];
    let mut token = String::new();
    let mut buffer = buffer.to_string();
    buffer.push('\n');
    for c in buffer.chars() {
        if c.is_whitespace() {
            if !token.is_empty() {
                tokens.push(token.clone());
                token.clear();
            }
        } else if c.is_ascii_punctuation() {
            if !token.is_empty() {
                tokens.push(token.clone());
                token.clear();
            }
            tokens.push(c.to_string());
        } else {
            token.push(c);
        }
    }
    tokens.push("\n".to_string());
    tokens
}

pub fn parse_element(tokens: &[String], pos: &mut usize, limit: i32, mut end_bracket: String) -> Option<element::Element> {
    if *pos >= tokens.len() {
        return None;
        //panic!("out of bounds");
    }

    let mut result: element::Element = if limit == -1 {
        // file level scope
        element::Element{
            value: element::Value::FileScope(),
            value_type: element::ValueType::None,
            childlen: Vec::new(),
        }
    } else {
        // read current token
        let token = &tokens[*pos];
        *pos += 1;
        if limit > 0 && token == "\n" {
            return parse_element(tokens, pos, limit, end_bracket);
        }
        // parse
        if let Some(el) = element::get_element(token) {
            el
        } else {
            return None;
        }
    };

    if let element::Value::Operator(_, priority) = result.value {
        // check priority of operator
        if priority < limit {
            *pos -= 1;
            return None;
        }
    } else if let element::Value::Bracket(ref bra) = result.value {
        // check end of bracket
        if *bra == end_bracket {
            return Some(result);
        } else {
            if let Some(eb) = element::get_end_bracket(bra) {
                end_bracket = eb;
            } else {
                panic!("Invalid syntax '{}' {}", bra, end_bracket);
            }
        }
    }

    // read childlen elements
    match result.value {
        element::Value::Operator(..) | element::Value::EndLine() => {
            return Some(result);
        }
        element::Value::Bracket(..) => {
            while let Some(next) = parse_element(tokens, pos, 0, end_bracket.clone()) {
                match next.value {
                    element::Value::Bracket(ref bra) => {
                        if *bra == end_bracket {
                            break;
                        } else {
                            result.childlen.push(next);
                        }
                    }
                    element::Value::EndLine() => {}
                    _ => {
                        result.childlen.push(next);
                    }
                }
            }
        }
        element::Value::FileScope() => {
            while let Some(next) = parse_element(tokens, pos, 0, end_bracket.clone()) {
                match next.value {
                    element::Value::EndLine() => {}
                    _ => {
                        result.childlen.push(next);
                    }
                }
            }
        }
        ref x if *x == element::get_symbol("let") => {
            if let Some(next) = parse_element(tokens, pos, 1, String::new()) {
                match next.value {
                    ref ope if *ope == element::get_operator("=") => {
                        result.childlen = next.childlen;
                    }
                    _ => {
                        panic!("Invalid syntax: operator 'let' can't found '=' token.");
                    }
                }
            }
        }
        _ => {}
    }

    // check if the next token is operator
    loop {
        if *pos >= tokens.len() {
            break;
        } else if let Some(next_element) = element::get_element(&tokens[*pos]) {
            if let element::Value::Operator(_, priority) = next_element.value {
                if priority < limit {  // check priority of operator
                    break;
                } else if let Some(next) = parse_element(tokens, pos, limit, end_bracket.clone()) {
                    result = reorder_elelemnt(tokens, pos, result, next, end_bracket.clone());
                } else {
                    panic!("Invalid syntax");
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return Some(result);
}

// use to parse operator
fn reorder_elelemnt(tokens: &[String], pos: &mut usize, mut el: element::Element, mut el_ope: element::Element, end_bracket: String) -> element::Element {
    if let element::Value::Operator(_, priority_ope) = el_ope.value {
        // finding left element
        if let element::Value::Operator(_, priority) = el.value {
            if priority_ope > priority {
                // el_ope is child node
                if let Some(el_right) = el.childlen.pop() {
                    let res = reorder_elelemnt(tokens, pos, el_right, el_ope, end_bracket);
                    el.childlen.push(res);
                    return el;
                } else {
                    panic!("Invalid syntax");
                }
            }
        }

        // el_ope is parent node. el is left node
        el_ope.childlen.push(el);
        // read right token
        let next = parse_element(tokens, pos, priority_ope, end_bracket);
        if let Some(next) = next {
            el_ope.childlen.push(next);
        } else {
            println!("Invalid syntax.");
        }
        return el_ope;
    } else {
        panic!("Invalid syntax");
    }
}
