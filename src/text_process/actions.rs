use super::stop_words::STOP_WORDS;
use regex::Regex;
use std::collections::HashMap;

/// Create a vector of tokens from a String of words.
pub fn tokenise(text: &str) -> Result<Vec<String>, String> {
    let words = text
        .split(' ')
        .filter(|s| !STOP_WORDS.contains(s) && !s.is_empty())
        .map(String::from)
        .collect::<Vec<String>>();

    if words.is_empty() {
        return Err(String::from("No words found."));
    }

    Ok(words)
}

/// Clean up the text in a string, removing certain 'code' elements.
pub fn clean_up_text(text: &str) -> String {
    let regex_strings = [
        "[^a-zA-Z ]", // single letters
        " +",         // more than one space
        r"\$.*\$",    // math fields
        // web links
        r"https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        // doi (?)
        r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        // code blocks
        r"```[a-z]*\n[\s\S]*?\n```",
        // yaml blocks
        r"---[a-z]*\n[\s\S]*?\n---",
    ];

    let mut output = String::from(text).to_lowercase();

    for reg in &regex_strings {
        output = match Regex::new(reg) {
            Ok(res) => res.replace_all(&output[..], " ").to_string(),
            Err(_) => output,
        };
    }

    output
}

///  Get a vector of all of the words in a corpus.
pub fn get_all_words(corpus: &[Vec<String>]) -> HashMap<String, i32> {
    let mut all_words: HashMap<String, i32> = HashMap::new();

    corpus
        .iter()
        .flatten()
        .map(|word| {
            let counter = all_words.entry(word.to_string()).or_insert(0);
            *counter += 1;
        })
        .for_each(drop);

    all_words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenise() {
        let text: String = "this is some text".to_string();

        let result_1: Result<Vec<String>, String> = tokenise(&text);

        assert_eq!(Ok(vec!["text".to_string()]), result_1);
    }

    #[test]
    fn test_tokenise_empty_input() {
        let empty_text: String = "".to_string();
        let result_2: Result<Vec<String>, String> = tokenise(&empty_text);

        assert_eq!(Err("No words found.".to_string()), result_2);
    }

    #[test]
    fn test_clean_up_text() {
        let text: String = "this is some text".to_string();

        let result = clean_up_text(&text);

        assert_eq!("this is some text".to_string(), result);
    }

    #[test]
    fn test_get_all_words() {
        let corpus: Vec<Vec<String>> = vec![
            vec![
                "this".to_string(),
                "is".to_string(),
                "some".to_string(),
                "text".to_string(),
            ],
            vec![
                "and".to_string(),
                "there".to_string(),
                "is".to_string(),
                "more".to_string(),
            ],
            vec!["this".to_string(), "too".to_string()],
        ];

        let mut words: HashMap<String, i32> = HashMap::new();
        words.insert("this".to_string(), 2);
        words.insert("is".to_string(), 2);
        words.insert("some".to_string(), 1);
        words.insert("text".to_string(), 1);
        words.insert("and".to_string(), 1);
        words.insert("there".to_string(), 1);
        words.insert("more".to_string(), 1);
        words.insert("too".to_string(), 1);

        let result: HashMap<String, i32> = get_all_words(&corpus);

        assert_eq!(words, result);
    }
}
