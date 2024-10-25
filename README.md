# lang2

### Compile source code

```
cargo run example/a
./build/main.out
```

### Run REPL
```
cargo run
```

## Install

### wasmtime
```
curl https://wasmtime.dev/install.sh -sSf | bash
```

#### run .wasm
```
wasmtime run build/main.wasm
```

### wasm-tools
```
cargo install wasm-tools
```

#### wasm -> wat
```
wasm-tools print build/main.wasm -o a.wat
```

#### wat -> wasm
```
wasm-tools parse a.wat -o a.wasm
```

### cargo-component
A cargo extension for authoring WebAssembly components
```
cargo install cargo-component
```

### wasi-sdk
https://github.com/WebAssembly/wasi-sdk/releases
```
mkdir misc
cd misc
wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-24/wasi-sdk-24.0-x86_64-linux.tar.gz
tar xvf wasi-sdk-24.0-x86_64-linux.tar.gz
cd ..
export WASI_SDK_PATH=./misc/wasi-sdk-24.0-x86_64-linux
```

## Test

```
cargo test -- --test-threads=1
```
```
cargo test backend::llvm::inkwell_example::hello_world -- --nocapture
```

## Docker

### Development
```
docker build . -t lang2-dev --target dev
docker run --rm -it -v .:/mnt lang2-dev
```

### Run test
```
docker build . -t lang2 --target test
```

### REPL
```
docker build . -t lang2 --target run
docker run --rm -it lang2
```
