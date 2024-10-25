FROM rust:1.82.0-bullseye AS dev

RUN apt-get update \
    && apt-get install -y lsb-release wget software-properties-common gnupg \
    && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 18 \
    && apt-get install -y libpolly-18-dev libzstd-dev

RUN mkdir -p /opt/wasi-sdk \
    && cd /opt/wasi-sdk \
    && wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-24/wasi-sdk-24.0-x86_64-linux.tar.gz \
    && tar xvf wasi-sdk-24.0-x86_64-linux.tar.gz \
    && rm wasi-sdk-24.0-x86_64-linux.tar.gz \
    && cd .. \
    && echo "export WASI_SDK_PATH=/opt/wasi-sdk/wasi-sdk-24.0-x86_64-linux" >> /etc/profile.d/wasi-sdk.sh

ENV WASI_SDK_PATH=/opt/wasi-sdk/wasi-sdk-24.0-x86_64-linux

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
