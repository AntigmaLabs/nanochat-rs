use pretty_assertions::assert_eq;
use rust_template::utils;
#[test]
fn test_integration_process_data() {
    let input = "hello world";
    let result = utils::process_data(input);
    assert_eq!(result, "HELLO WORLD");
}

#[test]
fn test_integration_validate_input() {
    assert!(utils::validate_input("valid input"));
    assert!(!utils::validate_input(""));

    let long_string = "a".repeat(101);
    assert!(!utils::validate_input(&long_string));
}
