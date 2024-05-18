use std::{error::Error, io::{self, Write}, path::{Path, PathBuf}};

pub fn link(objects: &Vec<String>) -> Result<PathBuf, io::Error> {
    let default_name = String::new();
    let name = objects.first().unwrap_or(&default_name);
    let mut exe_file = PathBuf::from(name);
    exe_file.set_extension("out");

    let output = std::process::Command::new("clang")
        .args(vec![
            name,
            "-v",
            "-o",
            exe_file.to_str().unwrap(),
        ])
        .output()?;

    println!("link executed: {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    // println!("{:?}", process.stdout);
    // println!("{}", String::from_utf8(process.stderr).unwrap());
    if !output.status.success() {
        panic!("link failed. status = {:?}", output.status.code());
        // anyhow::bail!("{}", String::from_utf8_lossy(&process.stderr));
    }

    Ok(exe_file)
}
