pub fn process_data(input: &str) -> String {
    input.to_uppercase()
}

pub fn validate_input(input: &str) -> bool {
    !input.is_empty() && input.len() <= 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data() {
        assert_eq!(process_data("hello"), "HELLO");
        assert_eq!(process_data(""), "");
    }

    #[test]
    fn test_validate_input() {
        assert!(validate_input("valid"));
        assert!(!validate_input(""));
        assert!(!validate_input(&"a".repeat(101)));
    }
}
