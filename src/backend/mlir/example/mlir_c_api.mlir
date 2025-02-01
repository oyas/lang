!MLIR_ptr = !llvm.struct<(!llvm.ptr)>

module attributes {llvm.data_layout = "e", llvm.target_triple = "x86_64-pc-linux-gnu"} {
	// "Hello, World!"
	memref.global "private" constant @string : memref<14xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x57,0x6f,0x72,0x6c,0x64,0x21,0]>

	// libc puts
	llvm.func external @puts(!llvm.ptr) -> ()

	// MlirContext mlirContextCreate(void);
	llvm.func external @mlirContextCreate() -> !MLIR_ptr
	// MlirLocation mlirLocationUnknownGet(MlirContext context);
	llvm.func external @mlirLocationUnknownGet(!MLIR_ptr) -> !MLIR_ptr
	// MlirModule mlirModuleCreateEmpty(MlirLocation location);
	llvm.func external @mlirModuleCreateEmpty(!MLIR_ptr) -> !MLIR_ptr
	// MlirOperation mlirModuleGetOperation(MlirModule module);
	llvm.func external @mlirModuleGetOperation(!MLIR_ptr) -> !MLIR_ptr
	// void mlirOperationDump(MlirOperation op);
	llvm.func external @mlirOperationDump(!MLIR_ptr) -> ()

	func.func @main() -> i64 {
		%c0_i64 = arith.constant 0 : i64
		%0 = memref.get_global @string : memref<14xi8>
		%1 = memref.extract_aligned_pointer_as_index %0 : memref<14xi8> -> index
		%2 = arith.index_cast %1 : index to i64
		%3 = llvm.inttoptr %2 : i64 to !llvm.ptr
		llvm.call @puts(%3) : (!llvm.ptr) -> ()

		%4 = llvm.call @mlirContextCreate() : () -> !MLIR_ptr  // MlirContext
		%5 = llvm.call @mlirLocationUnknownGet(%4) : (!MLIR_ptr) -> !MLIR_ptr  // MlirLocation
		%6 = llvm.call @mlirModuleCreateEmpty(%5) : (!MLIR_ptr) -> !MLIR_ptr  // MlirModule
		%7 = llvm.call @mlirModuleGetOperation(%6) : (!MLIR_ptr) -> !MLIR_ptr  // MlirOperation
		llvm.call @mlirOperationDump(%7) : (!MLIR_ptr) -> ()

		return %c0_i64 : i64
	}
}
