use std::cell::RefCell;
use std::{collections::HashMap, error::Error, rc::Weak, sync::Arc};
use std::fs;

use inkwell::values::BasicValueEnum;
use inkwell::{builder::Builder, context::Context, execution_engine::{ExecutionEngine, JitFunction}, module::Module, OptimizationLevel};

use crate::backend::llvm::{link, target, Triple};

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub modules: Vec<Arc<Module<'ctx>>>,
    pub current_module: Arc<Module<'ctx>>,
    pub builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
    pub values: RefCell<HashMap<String, BasicValueEnum<'ctx>>>,
}

pub fn new(context: &Context) -> Result<CodeGen<'_>, Box<dyn Error>> {
    // let context = Context::create();
    let current_module = Arc::new(context.create_module("main"));
    let execution_engine = current_module.create_jit_execution_engine(OptimizationLevel::None)?;
    let builder = context.create_builder();
    let mut modules = Vec::new();
    modules.push(Arc::clone(&current_module));
    let codegen = CodeGen {
        context,
        modules,
        current_module,
        builder,
        execution_engine,
        values: RefCell::new(HashMap::new()),
    };

    Ok(codegen)
}

impl<'ctx> CodeGen<'ctx> {
    pub fn create_module(&mut self, name: &str, source_file_name: &str) -> Arc<Module<'ctx>> {
        let module = self.context.create_module(name);
        module.set_source_file_name(source_file_name);

        self.execution_engine.add_module(&module).unwrap();

        let arc_module = Arc::new(module);
        self.modules.push(Arc::clone(&arc_module));
        self.current_module = Arc::clone(&arc_module);

        arc_module
    }

    pub fn get_main_module(&self) -> Arc<Module<'ctx>> {
        Arc::clone(self.modules.get(0).unwrap())
    }

    pub fn compile(&self) -> Result<(), Box<dyn Error>> {
        self.compile_with_target(&target::Triple::Default)
    }

    pub fn compile_with_target(&self, target_triple: &Triple) -> Result<(), Box<dyn Error>> {
        // Prepare build dir
        fs::create_dir_all("build").unwrap();

        // Write object files
        let mut objects = Vec::new();
        for module in &self.modules {
            let name = module.get_name().to_str().unwrap();
            // println!("----- Generated LLVM IR of {} -----", name);
            // println!("{}", module.to_string());
            // println!("----- End of LLVM IR of {} -----", name);
            target::run_passes_on(&self.get_main_module(), target_triple);
            objects.append(&mut target::write_to_file(module, target_triple, name));
        }
        match &target_triple {
            Triple::Wasm32WasiLibc => {
                println!("Add _start module.");
                let module = self.wasm_start_module();
                objects.append(&mut target::write_to_file(&module, target_triple, module.get_name().to_str().unwrap()));
            },
            _ => (),
        };
        println!("objects = {:?}", objects);

        // Link
        link::link(&objects, target_triple).unwrap();

        Ok(())
    }

    pub fn run_jit(&self) -> i32 {
        type MainFunc = unsafe extern "C" fn() -> i32;
        unsafe {
            let main: JitFunction<MainFunc> = self.execution_engine.get_function("main").unwrap();
            main.call()
        }
    }

    pub fn run_jit_eval_function(&self, fn_name: &str) -> String {
        type FuncType = unsafe extern "C" fn() -> *const u8;
        unsafe {
            let f: JitFunction<FuncType> = self.execution_engine.get_function(fn_name).unwrap();
            let result = f.call();
            let c_str = std::ffi::CStr::from_ptr(result as *const i8);
            String::from(c_str.to_str().unwrap())
        }
    }

    fn wasm_start_module(&self) -> Module<'ctx> {
        // _start function for wasm
        let module = self.context.create_module("_start");
        let builder = self.context.create_builder();
        let void_type = self.context.void_type();
        let i32_type = self.context.i32_type();
        let main_function_type = i32_type.fn_type(&[], false);
        let main_function = module.add_function("main", main_function_type, None);
        let start_function_type = void_type.fn_type(&[], false);
        let start_function = module.add_function("_start", start_function_type, None);
        let start_entry_block = self.context.append_basic_block(start_function, "entry");
        builder.position_at_end(start_entry_block);
        builder.build_call(main_function, &[], "start");
        builder.build_return(None).unwrap();
        println!("----- Generated LLVM IR of {} -----", module.get_name().to_str().unwrap());
        println!("{}", module.to_string());
        println!("----- End of LLVM IR of {} -----", module.get_name().to_str().unwrap());
        module
    }
}
