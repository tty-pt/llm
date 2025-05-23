#!/bin/sh
set -e

has_command() {
    command -v "$1" >/dev/null 2>&1
}

maybe_install() {
    to_install=""
    for pkg in "$@"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || \
		to_install="$to_install $pkg"
    done
    test ! -n "$to_install" || \
        sudo apt install -y $to_install
}

detect_gpu_backend_lshw() {
    maybe_install lshw
    lshw_out=$(sudo lshw -C display 2>/dev/null)

    echo "$lshw_out" | grep -iq 'driver=nvidia' && echo "nvidia" && return
    echo "$lshw_out" | grep -iq 'Intel Corporation' && echo "intel" && return
    echo "unknown"
}

detect_blas() {
    if ldconfig -p 2>/dev/null | grep -q libopenblas; then
        echo "openblas"
    elif ldconfig -p 2>/dev/null | grep -q libblas; then
        echo "blas"
    elif uname | grep -qi darwin && [ -f /System/Library/Frameworks/Accelerate.framework/Accelerate ]; then
        echo "accelerate"
    else
        echo "none"
    fi
}

build_with_cuda() {
    echo "🛠️ Building with CUDA..."
    cmake .. -DGGML_CUDA=on -DGGML_CUBLAS=on
    cmake --build . --config Release -- -j$(nproc)
}

build_cpu_only() {
    echo "🛠️ Building CPU-only..."
    cmake ..
    cmake --build . --config Release -- -j$(nproc)
}

maybe_install build-essential cmake git

GPU_BACKEND=$(detect_gpu_backend_lshw)
echo "🎯 GPU backend: $GPU_BACKEND"

BLAS_BACKEND=$(detect_blas)
echo "🧮 BLAS backend: $BLAS_BACKEND"

rm ../../.config.mk 2>/dev/null || true

if test "$GPU_BACKEND" = "nvidia"; then
    if ! has_command nvcc; then
        maybe_install nvidia-cuda-dev nvidia-cublas-dev
    fi
    build_with_cuda
    echo "CONFIG-cuda := y" >> ../../.config.mk
else
    build_cpu_only
fi


test "$BLAS_BACKEND" = "none" || echo "CONFIG-blas := y" >> ../../.config.mk
sudo make install

echo "🎉 Done!"

