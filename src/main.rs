mod calculations;
mod loading;
mod text_process;

use std::collections::HashMap;
use std::env;

/**
# Sesquioxide

A program to recursively load all of the text and markdown files in a directory and perform a
tf-idf calculation on them. This calculation is then used to compare command line input to all
of the documents and provide the file names of those which are most similar to the user input.
*/
fn main() {
    // Use the directory in which the program is run.
    let args: Vec<String> = env::args().collect();

    let dir_name = if args.len() == 2 { &args[1] } else { "." };

    let path_list: Vec<String> = match loading::load_paths(dir_name) {
        Ok(res) => res,
        Err(err) => {
            println!("{err}");
            std::process::exit(1)
        }
    };

    println!("Loading files.");

    // Loading data
    let corpus: Vec<Vec<String>> = match loading::load_corpus(&path_list) {
        Ok(res) => res,
        Err(err) => {
            println!("{err}");
            std::process::exit(1)
        }
    };

    let all_words: HashMap<String, i32> = text_process::get_all_words(&corpus);

    println!("Analysing corpus...");

    // Processing data
    let tf_vals: Vec<Vec<f64>> = calculations::tf_calculation(&corpus, &all_words);

    let idf_vals: Vec<f64> = calculations::idf_calculation(&corpus, &all_words);

    let tf_idf: Vec<Vec<f64>> = calculations::tf_idf_calculation(&tf_vals, &idf_vals);

    // Parse user input.
    loop {
        // Load the query
        println!("Search for: ");
        let mut line: String = String::new();
        let input = std::io::stdin().read_line(&mut line);

        if let Err(e) = input {
            println!("Error reading input. ({e})");
            continue;
        };

        // Process the input
        let cleaned_line: String = text_process::clean_up_text(&line);
        let test_input: Result<Vec<String>, _> = text_process::tokenise(&cleaned_line);

        match test_input {
            Ok(i) if i.contains(&String::from("quit")) => {
                println!("'quit' detected, ending program.");
                break;
            }

            Ok(i) => {
                if let Ok(res) = loading::process_input(&i, &all_words, &idf_vals) {
                    calculations::score_query(&res, &tf_idf, &path_list, 10);
                }
            }

            Err(_) => println!("Input error: stop words stripped from query."),
        };
    }
}
