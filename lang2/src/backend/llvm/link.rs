use std::{env, error::Error, io::{self, Write}, path::{Path, PathBuf}};

use super::Triple;

pub fn link(objects: &Vec<String>, target_triple: &Triple) -> Result<PathBuf, io::Error> {
    let default_name = String::new();
    let name = objects.first().unwrap_or(&default_name);
    let mut exe_file = PathBuf::from(name);
    exe_file.set_extension(match target_triple {
        Triple::Wasm32WasiLibc => "wasm",
        _ => "out"
    });

    let mut args = vec![
        // "-v",
        "-o",
        exe_file.to_str().unwrap(),
    ];
    let wasi_sdk_path = match &target_triple {
        Triple::Wasm32WasiLibc => match env::var("WASI_SDK_PATH") {
            Ok(val) => val,
            Err(err) => {
                println!("{}", err);
                panic!("Please set env variable 'WASI_SDK_PATH'.");
            },
        },
        _ => String::new(),
    };
    let lib_path = format!("-L{}/share/wasi-sysroot/lib/wasm32-wasi/", wasi_sdk_path);
    let wasm_ld_cmd = format!("{}/bin/wasm-ld", wasi_sdk_path);
    match &target_triple {
        // Triple::WASM32 => args.push("--target=wasm32"),
        Triple::Wasm32WasiLibc => {
            // args.push("--entry=main");
            // args.push("--no-entry");
            // args.push("--export-all");
            // args.push("--allow-undefined");
            args.push(&lib_path);
            args.push("-lc");
        },
        _ => (),
    };
    args.extend(objects.iter().map(|e| e.as_str()));
    // let output = std::process::Command::new("clang-17")
    let linker_cmd = match &target_triple {
        // Triple::Wasm32WasiLibc => "wasm-ld-17",
        Triple::Wasm32WasiLibc => &wasm_ld_cmd,
        _ => "cc",
        // _ => "clang-17",
    };
    println!("link command: {} {}", linker_cmd, args.join(" "));
    let output = std::process::Command::new(linker_cmd)
        .args(args)
        .output()?;

    println!("link executed: {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    println!("----- link stderr -----");
    io::stderr().write_all(&output.stderr).unwrap();
    println!("----- End of link stderr -----");
    if !output.status.success() {
        panic!("link failed. status = {:?}", output.status.code());
        // anyhow::bail!("{}", String::from_utf8_lossy(&process.stderr));
    }

    Ok(exe_file)
}
