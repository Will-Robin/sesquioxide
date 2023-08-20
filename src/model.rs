use super::calculations;
use super::text_process;
use std::collections::HashMap;

pub struct Model {
    pub tf: Vec<Vec<f64>>,
    pub idf: Vec<f64>,
    pub tf_idf: Vec<Vec<f64>>,
    pub words: HashMap<String, i32>,
}

impl Model {
    pub fn new(corpus: &[Vec<String>]) -> Self {
        let all_words: HashMap<String, i32> = text_process::get_all_words(corpus);

        let tf_vals: Vec<Vec<f64>> = calculations::tf_calculation(corpus, &all_words);

        let idf_vals: Vec<f64> = calculations::idf_calculation(corpus, &all_words);

        let tf_idf_vals: Vec<Vec<f64>> = calculations::tf_idf_calculation(&tf_vals, &idf_vals);

        Self {
            tf: tf_vals,
            idf: idf_vals,
            tf_idf: tf_idf_vals,
            words: all_words,
        }
    }
}
