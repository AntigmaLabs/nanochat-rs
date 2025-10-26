use rust_template::{init, utils};

fn main() {
    init();

    let samples = vec!["hello", "world", "rust", "template"];

    println!("Processing samples:");
    for sample in samples {
        if utils::validate_input(sample) {
            let processed = utils::process_data(sample);
            println!("  {} -> {}", sample, processed);
        }
    }
}
