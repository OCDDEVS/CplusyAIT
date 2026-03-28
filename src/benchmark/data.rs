use std::fs;
use std::collections::HashMap;

/// Loads and tokenizes the TinyShakespeare dataset (character level).
pub fn load_tinyshakespeare(filepath: &str) -> (Vec<usize>, usize) {
    let content = fs::read_to_string(filepath).unwrap_or_else(|_| {
        println!("Could not find {}, using synthetic fallback tokens.", filepath);
        "First citizen: Before we proceed any further, hear me speak.".to_string()
    });

    // Create char vocab
    let mut chars: Vec<char> = content.chars().collect();
    chars.sort();
    chars.dedup();
    let vocab_size = chars.len();

    let mut stoi: HashMap<char, usize> = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        stoi.insert(*c, i);
    }

    let tokens: Vec<usize> = content.chars().map(|c| *stoi.get(&c).unwrap()).collect();

    (tokens, vocab_size)
}
