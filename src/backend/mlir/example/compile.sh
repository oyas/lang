#!/bin/bash

SOURCE_FILE=${1:-hello.mlir}
LLVM_IR_FILE=${SOURCE_FILE%.*}.ll
OBJ_FILE=${SOURCE_FILE%.*}.o
OUTPUT_FILE=${SOURCE_FILE%.*}

mlir-opt --convert-to-llvm $SOURCE_FILE | mlir-translate --mlir-to-llvmir > $LLVM_IR_FILE
mlir-opt --convert-to-llvm $SOURCE_FILE | mlir-translate --mlir-to-llvmir | llc-19 --filetype=obj -o $OBJ_FILE --relocation-model=pic
clang-18 $OBJ_FILE -o $OUTPUT_FILE
