mod calculations;
mod loading;
mod model;
mod text_process;

use model::Model;
use std::env;

/**
# Sesquioxide

A program to recursively load all of the text and markdown files in a directory and perform a
tf-idf calculation on them. This calculation is then used to compare command line input to all
of the documents and provide the file names of those which are most similar to the user input.
*/
fn main() {
    // Use the directory supplied in arguments, otherwise use the directory in which the program is
    // run.
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

    println!("Creating model.");
    let model = Model::new(&corpus);

    loop {
        // Parse user input.
        println!("Search for: ");
        let mut line: String = String::new();
        let input = std::io::stdin().read_line(&mut line);

        if let Err(e) = input {
            println!("Error reading input, please try again. ({e})");
            continue;
        };

        // Process the input
        let cleaned_line: String = text_process::clean_up_text(&line);
        let test_input: Result<Vec<String>, _> = text_process::tokenise(&cleaned_line);

        match test_input {
            Ok(i) => {
                if i.contains(&String::from("quit")) {
                    println!("'quit' detected, ending program.");
                    break;
                };

                loading::process_input(&i, &model.words, &model).map_or_else(
                    |_| {
                        println!("Input processing failed.");
                    },
                    |res| {
                        calculations::score_query(&res, &model, &path_list, 10);
                    },
                );
            }

            Err(_) => println!("Error: no input (stop words were stripped from the query)"),
        };
    }
}
