!Tp = !llvm.struct<(i32,!llvm.ptr)>

module attributes {llvm.data_layout = "e", llvm.target_triple = "x86_64-pc-linux-gnu"} {
	memref.global "private" constant @string : memref<14xi8> = dense<[0x48,0x65,0x6c,0x6c,0x6f,0x2c,0x20,0x57,0x6f,0x72,0x6c,0x64,0x21,0]>
	memref.global "private" constant @s_ls : memref<8xi8> = dense<[0x2f,0x62,0x69,0x6e,0x2f,0x6c,0x73,0]>  // "/bin/ls"
	memref.global "private" constant @s_l : memref<3xi8> = dense<[0x2d,0x6c,0]>  // "-l"
	llvm.func external @puts(!llvm.ptr) -> ()
	llvm.func external @execve(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
	func.func @main() -> i64 {
		%c0_i64 = arith.constant 0 : i64
		%c1_i64 = arith.constant 1 : i64
		%c0_index = arith.constant 0 : index
		%c1_index = arith.constant 1 : index
		%c2_index = arith.constant 2 : index
		%c0_i8 = arith.constant 0 : i8
		%cb_i8 = arith.constant 0x62 : i8

		// puts("Hello, World!")
		%0 = memref.get_global @string : memref<14xi8>
		%1 = memref.extract_aligned_pointer_as_index %0 : memref<14xi8> -> index
		%2 = arith.index_cast %1 : index to i64
		%3 = llvm.inttoptr %2 : i64 to !llvm.ptr
		llvm.call @puts(%3) : (!llvm.ptr) -> ()

		// execve("/bin/ls", 0, 0)
		%4 = memref.get_global @s_ls : memref<8xi8>
		%5 = memref.extract_aligned_pointer_as_index %4 : memref<8xi8> -> index
		%6 = arith.index_cast %5 : index to i64
		%7 = llvm.inttoptr %6 : i64 to !llvm.ptr
		%8 = llvm.mlir.zero : !llvm.ptr
		// %10 = llvm.call @execve(%7, %8, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

		// execve("/bin/ls", ["-l", 0], 0)
		%11 = memref.alloca() : memref<3xindex>
		%12 = memref.get_global @s_l : memref<3xi8>
		%13 = memref.extract_aligned_pointer_as_index %12 : memref<3xi8> -> index
		memref.store %5, %11[%c0_index] : memref<3xindex>
		memref.store %13, %11[%c1_index] : memref<3xindex>
		memref.store %c0_index, %11[%c2_index] : memref<3xindex>
		%14 = memref.extract_aligned_pointer_as_index %11 : memref<3xindex> -> index
		%15 = arith.index_cast %14 : index to i64
		%16 = llvm.inttoptr %15 : i64 to !llvm.ptr
		%17 = llvm.call @execve(%7, %16, %8) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

		// alloca { i32, ptr }
		%20 = llvm.alloca %c1_i64 x !Tp : (i64) -> !llvm.ptr

		return %c0_i64 : i64
	}
}
