use super::model::Model;
use std::collections::HashMap;

/// Calculate the dot product of vec1 and vec2.
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2).map(|(&x, &y)| x * y).sum()
}

/// Determine the magnitude of a 1D vector.
pub fn vector_magnitude(vec1: &[f64]) -> f64 {
    vec1.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Determine the cosine similarity between vec1 and vec2.
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_prod: f64 = dot_product(vec1, vec2);

    if dot_prod == 0.0 {
        0.0
    } else {
        let mag_vec1: f64 = vector_magnitude(vec1);
        let mag_vec2: f64 = vector_magnitude(vec2);
        dot_prod / (mag_vec1 * mag_vec2)
    }
}

/// Generate TF values for each document.
pub fn tf_calculation(corpus: &[Vec<String>], all_words: &HashMap<String, i32>) -> Vec<Vec<f64>> {
    let mut tf_vals: Vec<Vec<f64>> = vec![vec![0.0; all_words.len()]; corpus.len()];

    for (i, item) in corpus.iter().enumerate() {
        for (j, (word, _)) in all_words.iter().enumerate() {
            tf_vals[i][j] = item
                .iter()
                .fold(0.0, |acc, x| if x == word { acc + 1.0 } else { acc });
        }
    }

    tf_vals
}

/// Perform an Inverse Document Frequency calculation.
pub fn idf_calculation(corpus: &[Vec<String>], all_words: &HashMap<String, i32>) -> Vec<f64> {
    let numerator: f64 = corpus.len() as f64;

    all_words
        .iter()
        .map(|(_, &x)| (numerator / (1.0 + f64::from(x))).ln())
        .collect::<Vec<f64>>()
}

/// Perform the text-frequency inverse document frequency calculation.
pub fn tf_idf_calculation(tf_vals: &[Vec<f64>], idf_vals: &[f64]) -> Vec<Vec<f64>> {
    // Generate TF-IDF values.
    let mut tf_idf: Vec<Vec<f64>> = vec![vec![0.0; tf_vals[0].len()]; tf_vals.len()];

    for (c1, doc) in tf_vals.iter().enumerate() {
        for (c2, word) in idf_vals.iter().enumerate() {
            tf_idf[c1][c2] = doc[c2] * word;
        }
    }

    tf_idf
}

/// Rank cosine similarities by index, remove entries with zero similarity.
fn rank_idx(cosine_similarities: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..cosine_similarities.len()).collect::<Vec<usize>>();

    indices.sort_by(|&a, &b| {
        cosine_similarities[b]
            .partial_cmp(&cosine_similarities[a])
            .expect("ranking failed")
    });

    indices
        .iter()
        .filter(|&&x| cosine_similarities[x] > 0.0)
        .copied()
        .collect()
}

pub fn score_query(test_vec: &[f64], model: &Model, path_list: &[String], top_n: usize) {
    let cos_sim: Vec<f64> = model
        .tf_idf
        .iter()
        .map(|x| cosine_similarity(test_vec, x))
        .collect::<Vec<f64>>();

    let ranking_idx = rank_idx(&cos_sim);

    let top = if ranking_idx.len() < top_n {
        ranking_idx.len()
    } else {
        top_n
    };

    println!("\nResults:\n");
    for &i in &ranking_idx[..top] {
        println!("{}, ({:.2})", path_list[i], cos_sim[i]);
    }

    println!("------");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let vector_a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let vector_b: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = dot_product(&vector_a, &vector_b);
        // note 16 decimal places for f64
        assert_eq!(result, 14.0000000000000000);
    }

    #[test]
    fn test_vector_magnitude() {
        let vector_a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = vector_magnitude(&vector_a);
        // note 16 decimal places for f64
        assert_eq!(result, 3.7416573867739413);
    }

    #[test]
    fn test_cosine_similarity_same() {
        let vector_a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let vector_b: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result: f64 = cosine_similarity(&vector_a, &vector_b);
        assert_eq!(result, 1.0000000000000000);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let vector_a: Vec<f64> = vec![4.0, 2.0, 3.0];
        let vector_b: Vec<f64> = vec![2.0, -4.0, 0.0];
        let result: f64 = cosine_similarity(&vector_a, &vector_b);
        assert_eq!(result, 0.0000000000000000);
    }

    #[test]
    fn test_tf_calculation() {
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

        let result = tf_calculation(&corpus, &words);

        // Collect line sums (result is irreproducible in inner vector ordering, stemming from
        // irreproducible HashMap iteration in tf_calculation())

        let test = result
            .iter()
            .map(|line| line.iter().sum())
            .collect::<Vec<f64>>();

        assert_eq!(
            test,
            vec![4.0000000000000000, 4.0000000000000000, 2.0000000000000000]
        );
    }

    #[test]
    fn test_idf_calculation() {
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

        let result = idf_calculation(&corpus, &words);

        let test: f64 = result.iter().sum();

        assert_eq!(test, 2.4327906486489868);
    }

    #[test]
    fn test_tf_idf_calculation() {
        let tf_vals: Vec<Vec<f64>> = vec![vec![1.0, 1.0, 1.0]];
        let idf_vals: Vec<f64> = vec![1.0];

        let result: Vec<Vec<f64>> = tf_idf_calculation(&tf_vals, &idf_vals);

        assert_eq!(result, vec![vec![1.0, 0.0, 0.0]]);
    }

    #[test]
    fn test_rank_idx() {
        let values = [1.0, 3.0, 2.0, 6.0, 5.0, 0.0, 0.0, 0.0];

        let result = rank_idx(&values);

        assert_eq!(result, vec![3, 4, 1, 2, 0]);
    }
}
