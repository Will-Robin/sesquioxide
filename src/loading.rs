use super::text_process;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

fn walk_directory(dir_name: &str) -> Vec<String> {
    let mut path_list: Vec<String> = Vec::new();
    for entry in WalkDir::new(dir_name)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir())
    {
        let f_name: String = String::from(entry.file_name().to_string_lossy());

        let path_name = Path::new(&f_name);
        if path_name
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("md"))
            || path_name
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("txt"))
        {
            if let Some(res) = entry.path().to_str() {
                path_list.push(res.to_string());
            }
        }
    }

    path_list
}

/// Load the paths of markdown or .txt files below a directory.
pub fn load_paths(dir_name: &str) -> Result<Vec<String>, &str> {
    if Path::new(dir_name).is_dir() {
        let path_list = walk_directory(dir_name);
        if path_list.is_empty() {
            Err("No files found in directory.")
        } else {
            Ok(path_list)
        }
    } else {
        Err("Directory does not exist.")
    }
}

/// Load a corpus from a list of paths.
pub fn load_corpus(path_list: &[String]) -> Result<Vec<Vec<String>>, &str> {
    let mut corpus: Vec<Vec<String>> = Vec::new();

    for path_string in path_list {
        if let Ok(res) = extract_contents(path_string) {
            corpus.push(res);
        };
    }

    if corpus.is_empty() {
        Err("No words extracted from files.")
    } else {
        Ok(corpus)
    }
}

/// Load the contents of a file into a tokenised vector.
pub fn extract_contents(path: &str) -> Result<Vec<String>, String> {
    let contents: String = fs::read_to_string(path).map_or_else(|_| String::new(), |res| res);

    let cleaned_text = text_process::clean_up_text(&contents);

    let words: Result<Vec<String>, String> = text_process::tokenise(&cleaned_text);

    match words {
        Ok(words) if words.is_empty() => Err(String::from("Zero len vector.")),
        Ok(words) => Ok(words),
        Err(message) => Err(message),
    }
}

/// Process user input and a tf-idf vector from it.
pub fn process_input(
    test_input: &[String],
    all_words: &HashMap<String, i32>,
    idf_vals: &[f64],
) -> Result<Vec<f64>, String> {
    if test_input.is_empty() {
        return Err(String::from("Empty query"));
    }

    let mut test_vec: Vec<f64> = vec![0.0; idf_vals.len()];

    // Create a vector from the input (only consider words from the input which are already in the
    // corpus, and use idf values as calculated for the corpus).
    for (c, (word, _count)) in all_words.iter().enumerate() {
        if test_input.contains(word) {
            test_vec[c] += 1.0 * idf_vals[c];
        }
    }

    Ok(test_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_paths_ok() {
        let result: Result<Vec<String>, &str> = load_paths("data");

        // This assertion may fail if the system serves files in a different order to the vector
        // given below.
        assert_eq!(
            Ok(vec![
                "data/doc4.txt".to_string(),
                "data/doc1.txt".to_string(),
                "data/doc3.txt".to_string(),
                "data/doc2.txt".to_string()
            ]),
            result
        );
    }

    #[test]
    fn test_load_paths_error() {
        // test an invalid path to a file.
        let error_res: Result<Vec<String>, &str> = load_paths("wrong_path");
        assert_eq!(Err("Directory does not exist."), error_res);
    }

    #[test]
    fn test_load_corpus_ok() {
        let paths: Vec<String> = vec!["data/doc1.txt".to_string()];
        let result: Result<Vec<Vec<String>>, &str> = load_corpus(&paths);

        assert_eq!(
            Ok(vec![vec!["sky".to_string(), "blue".to_string()]]),
            result
        );
    }

    #[test]
    fn test_load_corpus_error() {
        let paths: Vec<String> = vec!["abcdefg.txt".to_string()];
        let result: Result<Vec<Vec<String>>, &str> = load_corpus(&paths);

        assert_eq!(Err("No words extracted from files."), result);
    }

    #[test]
    fn test_extract_contents_ok() {
        let result: Result<Vec<String>, String> = extract_contents("data/doc1.txt");

        assert_eq!(Ok(vec!["sky".to_string(), "blue".to_string()]), result);

        let error_res: Result<Vec<String>, String> = extract_contents("");

        assert_eq!(Err(String::from("No words found.")), error_res);
    }

    #[test]
    fn test_extract_contents_error() {
        let result: Result<Vec<String>, String> = extract_contents("abcdefg.txt");

        assert_eq!(Err(String::from("No words found.")), result);
    }

    #[test]
    fn test_process_input_ok() {
        let test_input: Vec<String> = vec![
            "this is some text".to_string(),
            "and there is more".to_string(),
            "this too".to_string(),
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

        let idf_vals: Vec<f64> = vec![1.0];

        let result: Result<Vec<f64>, String> = process_input(&test_input, &words, &idf_vals);

        assert_eq!(Ok(vec![0.0000000000000000]), result);
    }

    #[test]
    fn test_process_input_error() {
        let mut words: HashMap<String, i32> = HashMap::new();
        words.insert("this".to_string(), 2);
        words.insert("is".to_string(), 2);
        words.insert("some".to_string(), 1);
        words.insert("text".to_string(), 1);
        words.insert("and".to_string(), 1);
        words.insert("there".to_string(), 1);
        words.insert("more".to_string(), 1);
        words.insert("too".to_string(), 1);

        let idf_vals: Vec<f64> = vec![1.0];

        // test empty query
        let error_res: Result<Vec<f64>, String> = process_input(&[], &words, &idf_vals);

        assert_eq!(Err(String::from("Empty query")), error_res);
    }
}
