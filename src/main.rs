use rust_template::{init, utils};
use std::env;

fn main() {
    init();

    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let input = &args[1];
        if utils::validate_input(input) {
            let processed = utils::process_data(input);
            println!("Processed: {}", processed);
        } else {
            eprintln!("Invalid input");
            std::process::exit(1);
        }
    } else {
        println!("Welcome to Rust Template!");
        println!("Usage: {} <input>", args[0]);
    }
}
