use std::path::Path;

use inkwell::{llvm_sys::target_machine, module::Module, passes::PassBuilderOptions, support::LLVMString, targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple}, OptimizationLevel};


pub enum Triple {
    Default,
    Wasm32WasiLibc,
    Triple(String),
}

pub fn get_target_machine(target_triple: &Triple) -> TargetMachine {
    match target_triple {
        Triple::Wasm32WasiLibc => get_target_machine_wasm32(),
        Triple::Default => get_target_machine_default(),
        Triple::Triple(_) => todo!(),
    }
}

pub fn run_passes_on(
    module: &Module,
    target_triple: &Triple,
) {
    Target::initialize_all(&InitializationConfig::default());
    let target_machine = get_target_machine(target_triple);

    let passes: &[&str] = &[
        "instcombine",
        "reassociate",
        "gvn",
        "simplifycfg",
        // "basic-aa",
        "mem2reg",
    ];

    // module
    //     .run_passes(passes.join(",").as_str(), &target_machine, PassBuilderOptions::create())
    //     .unwrap();

    // let path = Path::new("main.asm");
    // target_machine.write_to_file(&module, FileType::Assembly, &path);

    // let path = Path::new("main.o");
    // target_machine.write_to_file(&module, FileType::Object, &path);

    module
        .run_passes(passes.join(",").as_str(), &target_machine, PassBuilderOptions::create())
        .unwrap();

    // let path = Path::new("main-wasm.asm");
    // target_machine_wasm.write_to_file(&module, FileType::Assembly, &path);

    // let path = Path::new("main-wasm.o");
    // target_machine_wasm.write_to_file(&module, FileType::Object, &path);
}

// write 4 file
// - object file : {build_dir}/{file_name}.o
// - asm file    : {build_dir}/{file_name}.asm
// - bc file     : {build_dir}/{file_name}.bc
// - ll file     : {build_dir}/{file_name}.ll
pub fn write_to_file(
    module: &Module,
    target_triple: &Triple,
    name: &str,
) -> Vec<String> {
    let target_machine = get_target_machine(target_triple);

    let build_dir = "build";

    let path_str = format!("{build_dir}/{name}.asm");
    let path = Path::new(&path_str);
    target_machine.write_to_file(module, FileType::Assembly, &path).unwrap();

    let path_str_object = format!("{build_dir}/{name}.o");
    let path = Path::new(&path_str_object);
    target_machine.write_to_file(module, FileType::Object, &path).unwrap();

    let path_str = format!("{build_dir}/{name}.bc");
    let path = Path::new(&path_str);
    if !module.write_bitcode_to_path(path) {
        panic!("faild to write_bitcode_to_path");
    }

    let path_str = format!("{build_dir}/{name}.ll");
    let path = Path::new(&path_str);
    module.print_to_file(path).unwrap();

    // object files
    vec![path_str_object]
}

fn get_target_machine_wasm32() -> TargetMachine {
    let wasm_triple = TargetTriple::create("wasm32-unknown-unknown");
    let target_wasm = Target::from_triple(&wasm_triple).unwrap();
    let target_machine_wasm = target_wasm
        .create_target_machine(
            &wasm_triple,
            "generic",
            "",
            OptimizationLevel::None,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .unwrap();

    target_machine_wasm
}

fn get_target_machine_default() -> TargetMachine {
    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple).unwrap();
    let target_machine = target
        .create_target_machine(
            &target_triple,
            "generic",
            "",
            OptimizationLevel::None,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .unwrap();

    target_machine
}