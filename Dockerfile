FROM rust:1.84.0-bullseye AS dev

ENV LLVM_VERSION=19

RUN apt-get update \
    && apt-get install -y lsb-release wget software-properties-common gnupg \
    && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- $LLVM_VERSION \
    && apt-get install -y libpolly-${LLVM_VERSION}-dev libzstd-dev libmlir-${LLVM_VERSION}-dev \
    && ln -s /usr/lib/llvm-${LLVM_VERSION}/bin/llvm-config /usr/local/bin/llvm-config

ENV WASI_VERSION=25

RUN mkdir -p /opt/wasi-sdk \
    && cd /opt/wasi-sdk \
    && wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_VERSION}/wasi-sdk-${WASI_VERSION}.0-x86_64-linux.tar.gz \
    && tar xvf wasi-sdk-${WASI_VERSION}.0-x86_64-linux.tar.gz \
    && rm wasi-sdk-${WASI_VERSION}.0-x86_64-linux.tar.gz \
    && cd .. \
    && echo "export WASI_SDK_PATH=/opt/wasi-sdk/wasi-sdk-${WASI_VERSION}.0-x86_64-linux" >> /etc/profile.d/wasi-sdk.sh

RUN curl https://wasmtime.dev/install.sh -sSf | bash

RUN cargo install --locked wasm-tools

ENV WASI_SDK_PATH=/opt/wasi-sdk/wasi-sdk-${WASI_VERSION}.0-x86_64-linux

WORKDIR /mnt

CMD ["bash"]


FROM dev AS test

WORKDIR /usr/local/src/lang
COPY . .

RUN cargo test -- --test-threads=1 \
    && cargo clean


FROM test AS run

RUN cargo build --release \
    && cp /usr/local/src/lang/target/release/lang2 /usr/local/bin/lang2 \
    && cargo clean

ENTRYPOINT ["/usr/local/bin/lang2"]
