#!/bin/bash

SOURCE_FILE=${1:-mlir_c_api.mlir}
LLVM_IR_FILE=${SOURCE_FILE%.*}.ll
OBJ_FILE=${SOURCE_FILE%.*}.o
OUTPUT_FILE=${SOURCE_FILE%.*}

OPT_OPTIONS="--lower-affine --convert-scf-to-cf --convert-to-llvm"


# compile
mlir-opt $OPT_OPTIONS $SOURCE_FILE
if [[ $? != 0 ]]; then
	exit $?
fi
mlir-opt $OPT_OPTIONS $SOURCE_FILE | mlir-translate --mlir-to-llvmir | llc-19 --filetype=obj -o $OBJ_FILE --relocation-model=pic

# link
bash link-mlir-lib.sh $OBJ_FILE -o $OUTPUT_FILE

# execute
LD_LIBRARY_PATH=`llvm-config --libdir` ./$OUTPUT_FILE
