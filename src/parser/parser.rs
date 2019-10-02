use super::element;

pub fn line_to_tokens(buffer: &str) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];
    let mut token = String::new();
    let mut buffer = buffer.to_string();
    buffer.push('\n');
    for c in buffer.chars() {
        if c.is_whitespace() {
            if let Some(x) = token.chars().nth(0) {
                if !x.is_whitespace() {
                    tokens.push(token.clone());
                    token.clear();
                }
            }
            token.push(c);
        } else if c.is_ascii_punctuation() {
            if !token.is_empty() {
                tokens.push(token.clone());
                token.clear();
            }
            if c == '#' {
                // following tokens are comment
                break;
            } else {
                tokens.push(c.to_string());
                continue;
            }
        } else {
            if let Some(x) = token.chars().nth(0) {
                if x.is_whitespace() {
                    tokens.push(token.clone());
                    token.clear();
                }
            }
            token.push(c);
        }
    }
    tokens.push("\n".to_string());
    tokens
}

pub fn parse_element(
    tokens: &[String],
    pos: &mut usize,
    expect: &str,
    limit: i32,
    end_bracket: &str,
    read_one_element: bool,
) -> Option<element::Element> {
    // check pos is within range
    if *pos >= tokens.len() {
        return None;
        //panic!("out of bounds");
    }

    let mut end_bracket = end_bracket.to_string();

    // current element
    let mut result: element::Element = if limit == -1 {
        // file level scope
        element::Element::new(element::Value::FileScope())
    } else {
        // read current token
        let token = if expect.is_empty() {
            *pos += 1;
            tokens[*pos - 1].to_string()
        } else {
            let mut token = String::new();
            let mut cur = *pos;
            while token.len() < expect.len() {
                if let Some(el) = element::get_element(&tokens[cur]) {
                    match el.value {
                        element::Value::EndLine() => {}
                        element::Value::Space(..) => {}
                        _ => token += &tokens[cur],
                    }
                }
                cur += 1;
            }
            if token == expect {
                *pos = cur;
            } else {
                return None;
            }
            token
        };
        // parse
        if let Some(el) = element::get_element(&token) {
            match el.value {
                element::Value::EndLine() => {
                    if limit > 0 {
                        return parse_element(tokens, pos, "", limit, &end_bracket, false);
                    } else {
                        el
                    }
                }
                element::Value::Space(..) => {
                    return parse_element(tokens, pos, "", limit, &end_bracket, false);
                }
                _ => el,
            }
        } else {
            return None;
        }
    };

    // element specific process
    match result.value {
        element::Value::Operator(_, priority) => {
            // check priority of operator
            if priority < limit {
                *pos -= 1;
                return None;
            }
        }
        element::Value::Bracket(ref bra) => {
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
        _ => {}
    }

    // read childlen elements
    match &result.value {
        element::Value::Identifier(_) => {
            if let Some(next) = element::get_element(&tokens[*pos]) {
                if next.value == element::get_bracket("(") {
                    match parse_element(tokens, pos, "", 0, &end_bracket, true) {
                        Some(next_el) => {
                            println!("{}", next_el);
                            result = element::make_function_call(result, next_el)
                        }
                        None => panic!("Invalid syntax"),
                    }
                }
            }
        }
        element::Value::Operator(..) | element::Value::EndLine() => {
            return Some(result);
        }
        element::Value::Bracket(bra) if bra == "{" => {
            while let Some(next) = parse_element(tokens, pos, "", 0, &end_bracket, false) {
                match next.value {
                    element::Value::Bracket(ref bra) if *bra == end_bracket => break,
                    element::Value::EndLine() => {}
                    _ => result.childlen.push(next),
                }
            }
        }
        element::Value::Bracket(bra) if bra == "(" => {
            let mut next_is_comma = false;
            while let Some(next) = parse_element(tokens, pos, "", 1, &end_bracket, false) {
                match next.value {
                    element::Value::Bracket(ref bra) if *bra == end_bracket => break,
                    element::Value::Comma() => {
                        assert!(next_is_comma, "Invalid syntax: next is not comma");
                        next_is_comma = !next_is_comma
                    }
                    _ => {
                        assert!(!next_is_comma, "Invalid syntax: next is comma");
                        result.childlen.push(next)
                    }
                }
            }
        }
        element::Value::FileScope() => {
            let mut next_is_endline = false;
            while let Some(next) = parse_element(tokens, pos, "", 0, &end_bracket, false) {
                match next.value {
                    element::Value::EndLine() => {
                        next_is_endline = false;
                    }
                    _ => {
                        assert!(
                            !next_is_endline,
                            format!("Invalid syntax: unexpected element {:?}.", next.value)
                        );
                        result.childlen.push(next);
                        next_is_endline = true;
                    }
                }
            }
        }
        x if *x == element::get_symbol("let") => match parse_element(tokens, pos, "", 1, "", false) {
            Some(next) => match next.value {
                ref ope if *ope == element::get_operator("=") => {
                    result.childlen = next.childlen;
                }
                _ => {
                    panic!("Invalid syntax: operator 'let' couldn't find '=' token.");
                }
            },
            None => {
                panic!("Invalid syntax: operator 'let' couldn't find '=' token.");
            }
        },
        x if *x == element::get_symbol("if") => {
            match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => result.childlen.push(next),
                None => panic!("Invalid syntax: operator 'if' couldn't find statement."),
            }
            match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => match &next.value {
                    ope if *ope == element::get_bracket("{") => {
                        result.childlen.push(next);
                    }
                    _ => panic!("Invalid syntax: operator 'if' couldn't find '{' token."),
                },
                None => panic!("Invalid syntax: operator 'if' couldn't find '{' token."),
            }
            if let Some(next_element) = element::get_next_nonblank_element(tokens, *pos) {
                if next_element.value == element::get_symbol("else") {
                    match parse_element(tokens, pos, "else", 1, "", false) {
                        Some(next) => result.childlen.push(next.childlen.first().unwrap().clone()),
                        None => panic!("Invalid syntax: reading 'else'"),
                    }
                }
            }
        }
        x if *x == element::get_symbol("else") => match parse_element(tokens, pos, "", 1, "", false) {
            Some(next) => result.childlen.push(next),
            None => panic!("Invalid syntax: operator 'else' couldn't find next element."),
        },
        x if *x == element::get_symbol("for") => {
            match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => result.childlen.push(next),
                None => panic!("Invalid syntax: operator 'for' couldn't find statement."),
            }
            match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => match &next.value {
                    ope if *ope == element::get_bracket("{") => result.childlen.push(next),
                    _ => panic!("Invalid syntax: operator 'for' couldn't find '{' token."),
                },
                None => panic!("Invalid syntax: operator 'for' couldn't find '{' token."),
            }
        }
        x if *x == element::get_symbol("fun") => {
            let fcall = match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => match &next.value {
                    element::Value::FunctionCall(_) => next,
                    _ => panic!("Invalid syntax: symbol 'fun' couldn't parse function."),
                },
                None => panic!("Invalid syntax: symbol 'fun' couldn't parse function."),
            };
            let body = match parse_element(tokens, pos, "", 1, "", false) {
                Some(next) => match &next.value {
                    ope if *ope == element::get_bracket("{") => next,
                    _ => panic!("Invalid syntax: operator 'fun' couldn't find '{' token."),
                },
                None => panic!("Invalid syntax: symbol 'fun' couldn't find '{' token."),
            };
            // make let element
            result = element::make_function(result, fcall, body);
        }
        _ => {}
    }

    // check if the next token is operator
    if !read_one_element {
        loop {
            if *pos >= tokens.len() {
                break;
            } else if let Some(next_element) = element::get_next_operator(tokens, *pos) {
                if let element::Value::Operator(ope, priority) = next_element.value {
                    if priority < limit {
                        // check priority of operator
                        break;
                    }
                    match parse_element(tokens, pos, &ope, priority, &end_bracket, false) {
                        Some(next) => {
                            result = reorder_elelemnt(tokens, pos, result, next, &end_bracket)
                        }
                        None => panic!("Invalid syntax"),
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    return Some(result);
}

// use to parse operator
fn reorder_elelemnt(
    tokens: &[String],
    pos: &mut usize,
    mut el: element::Element,
    mut el_ope: element::Element,
    end_bracket: &str,
) -> element::Element {
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
        let next = parse_element(tokens, pos, "", priority_ope, end_bracket, false);
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
