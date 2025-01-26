module attributes {llvm.data_layout = "e", llvm.target_triple = "x86_64-pc-linux-gnu"} {
	memref.global "private" constant @s_ok : memref<3xi8> = dense<[0x4f,0x4b,0]>  // "OK"
	memref.global "private" constant @s_ng : memref<3xi8> = dense<[0x4e,0x47,0]>  // "NG"
	llvm.func external @puts(!llvm.ptr) -> ()
	func.func @main() -> i64 {
		%c0_i64 = arith.constant 0 : i64
		%c1_i64 = arith.constant 1 : i64
		%c0_index = arith.constant 0 : index
		%c1_index = arith.constant 1 : index
		%c2_index = arith.constant 2 : index
		%c0_i8 = arith.constant 0 : i8
		%cb_i8 = arith.constant 0x62 : i8
		%c0_ptr = llvm.mlir.zero : !llvm.ptr

		// "OK"
		%4 = memref.get_global @s_ok : memref<3xi8>
		%5 = memref.extract_aligned_pointer_as_index %4 : memref<3xi8> -> index
		%6 = arith.index_cast %5 : index to i64
		%sp_ok = llvm.inttoptr %6 : i64 to !llvm.ptr
		// "NG"
		%7 = memref.get_global @s_ng : memref<3xi8>
		%8 = memref.extract_aligned_pointer_as_index %7 : memref<3xi8> -> index
		%9 = arith.index_cast %8 : index to i64
		%sp_ng = llvm.inttoptr %9 : i64 to !llvm.ptr

		// for i in 0..5 if (i == 3) puts("NG") else puts("OK")
		affine.for %i = 0 to 5 {
			affine.if affine_set<(d0) : (d0 == 3)>(%i) {
				llvm.call @puts(%sp_ng) : (!llvm.ptr) -> ()
			} else {
				llvm.call @puts(%sp_ok) : (!llvm.ptr) -> ()
			}
		}

		// for i in 0..2 if (i == 0) puts("NG") else puts("OK")
		scf.for %j = %c0_index to %c2_index step %c1_index {
			%b = arith.cmpi "eq", %j, %c0_index : index
			scf.if %b {
				llvm.call @puts(%sp_ng) : (!llvm.ptr) -> ()
			} else {
				llvm.call @puts(%sp_ok) : (!llvm.ptr) -> ()
			}
		}

		return %c0_i64 : i64
	}
}
