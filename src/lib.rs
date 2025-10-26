pub mod error;
pub mod utils;

pub use error::{Error, Result};

pub fn init() {
    println!("Library initialized");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
