pub fn line_to_tokens(line: &str) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];
    let mut token = String::new();
    let mut buffer = line.to_string();
    if !buffer.ends_with('\n') {
        buffer.push('\n');
    }
    for c in buffer.chars() {
        if c.is_whitespace() {
            if let Some(x) = token.chars().nth(0) {
                if !x.is_whitespace() {
                    tokens.push(token.clone());
                    token.clear();
                }
            }
            token.push(c);
        } else if c.is_ascii_punctuation() || c == '\n' {
            if !token.is_empty() {
                tokens.push(token.clone());
                token.clear();
            }
            tokens.push(c.to_string());
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
    if !token.is_empty() {
        tokens.push(token.clone());
    }
    tokens
}
