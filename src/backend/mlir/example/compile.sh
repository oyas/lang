#!/bin/bash

SOURCE_FILE=${1:-hello.mlir}
LLVM_IR_FILE=${SOURCE_FILE%.*}.ll
OBJ_FILE=${SOURCE_FILE%.*}.o
OUTPUT_FILE=${SOURCE_FILE%.*}

OPT_OPTIONS="--lower-affine --convert-scf-to-cf --convert-to-llvm"

mlir-opt $OPT_OPTIONS $SOURCE_FILE
if [[ $? != 0 ]]; then
	exit $?
fi
mlir-opt $OPT_OPTIONS $SOURCE_FILE | mlir-translate --mlir-to-llvmir > $LLVM_IR_FILE
mlir-opt $OPT_OPTIONS $SOURCE_FILE | mlir-translate --mlir-to-llvmir | llc-19 --filetype=obj -o $OBJ_FILE --relocation-model=pic
clang-18 $OBJ_FILE -o $OUTPUT_FILE
