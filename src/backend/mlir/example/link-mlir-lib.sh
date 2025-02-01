#!/bin/bash


libdir=`llvm-config --link-static --libdir`
libdir_libs=`ls /usr/lib/llvm-19/lib | grep -E "lib[^ ]+\.a" | sed -e 's/lib\([^ ]*\)\.a/-l\1/g'`
mlir="-lMLIR"
libnames=`llvm-config --link-static --libnames | sed -e 's/lib\([^ ]*\)\.a/-l\1/g'`
system_libs=`llvm-config --link-static --system-libs`
libcpp="-lstdc++"

set -x

clang $@ -L$libdir $libdir_libs $mlir $libnames $system_libs $libcpp  #-static -static-libsan -static-pie
