use std::{collections::HashMap, ffi::CStr, fs, ops::{Deref, DerefMut}, sync::{Arc, Mutex, RwLock, RwLockReadGuard}};

use inkwell::llvm_sys;
use melior::{
    dialect::{arith, func, DialectRegistry}, ir::{attribute::{StringAttribute, TypeAttribute}, r#type::FunctionType, *}, pass, utility::register_all_dialects, Context, Error, ExecutionEngine
};
use mlir_sys::{mlirTranslateModuleToLLVMIR, LLVMOpaqueContext};

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Arc<RwLock<Module<'ctx>>>,
    pub modules: Vec<Arc<RwLock<Module<'ctx>>>>,
    pub symbols: HashMap<String, SymbolInfo>,
}

pub struct SymbolInfo {
    pub name: String,
    pub ptr: *mut (),
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let module = Arc::new(RwLock::new(Module::new(Location::unknown(&context))));
        let modules = vec![Arc::clone(&module)];
        CodeGen {
            context,
            module,
            modules,
            symbols: HashMap::new(),
        }
    }

    pub fn new_module(&mut self) {
        self.module = Arc::new(RwLock::new(Module::new(Location::unknown(&self.context))));
        self.modules.push(Arc::clone(&self.module));
    }

    pub fn module(&self) -> RwLockReadGuard<Module<'ctx>> {
        self.module.read().unwrap()
    }

    pub fn run_pass(&mut self) {
        let pass_manager = pass::PassManager::new(&self.context);
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        let mut module = self.module.write().unwrap();
        assert_eq!(pass_manager.run(&mut module), Ok(()));
    }

    pub fn create_engine(&self) -> ExecutionEngine {
        let engine = ExecutionEngine::new(&self.module.read().unwrap(), 2, &[], true);
        let mut result: usize = 0;
        for (name, symbol) in &self.symbols {
            unsafe {
                engine.register_symbol(name, symbol.ptr);
            };
        }
        engine
    }

    pub fn execute_no_args(&self, name: &str) -> Result<usize, Error> {
        let engine = self.create_engine();
        let mut result: usize = 0;
        let res = unsafe {
            engine.invoke_packed(
                name,
                &mut [
                    &mut result as *mut usize as *mut (),
                ],
            )
        };
        res.map(|_| result)
    }

    pub fn to_inkwell_module(self, inkwell_context: &inkwell::context::Context) -> inkwell::module::Module {
        let module = self.module.read().unwrap();
        unsafe {
            let m = mlirTranslateModuleToLLVMIR(
                module.as_operation().to_raw(),
                inkwell_context.raw() as *mut LLVMOpaqueContext
            );
            inkwell::module::Module::new(m as *mut inkwell::llvm_sys::LLVMModule)
        }
    }

    pub fn save_llvm_ir(&self, path: &str, show_debug: bool) {
        let engine = self.create_engine();
        let module = self.module.read().unwrap();
        let llvm_module = unsafe {
            let llvm_context = llvm_sys::core::LLVMContextCreate();
            let llvm_module = mlirTranslateModuleToLLVMIR(
                module.as_operation().to_raw(),
                llvm_context as *mut LLVMOpaqueContext,
            );
            let ir_str = llvm_sys::core::LLVMPrintModuleToString(llvm_module as *mut llvm_sys::LLVMModule);
            CStr::from_ptr(ir_str).to_str().unwrap()
        };
        if show_debug {
            println!("----- Generated LLVM IR ({}) -----", path);
            println!("{}", llvm_module);
            println!("----- End of LLVM IR -----");
        }
        if !path.is_empty() {
            fs::write(path, llvm_module.to_string()).unwrap();
        }
    }

    pub fn save_object(&self, path: &str) {
        let engine = self.create_engine();
        engine.dump_to_object_file(path);
    }
}
