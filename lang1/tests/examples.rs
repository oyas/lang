use rustlang::run;
use std::fs;

#[test]
fn run_examples() {
    let dir_name = "./examples";
    for path in fs::read_dir(dir_name).unwrap() {
        let file_path = path.unwrap().path().display().to_string();
        let result = run(&file_path, false);
        assert!(result.is_some(), format!("Failed to run {}.", file_path));
    }
}
