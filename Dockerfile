# Stage 1: Build the framework using Arch Linux (Rolling Release)
FROM archlinux:latest as builder

# Install Rust toolchain, C++ compilers, and standard build tools via pacman
RUN pacman -Syu --noconfirm && pacman -S --noconfirm \
    base-devel \
    cmake \
    gcc \
    rust \
    cargo \
    openssl \
    pkgconf

WORKDIR /usr/src/app

# Copy manifests to cache dependencies
COPY Cargo.toml Cargo.lock ./
# Create dummy source files so we can compile dependencies (layer caching)
RUN mkdir src kernels && \
    echo "fn main() {}" > src/main.rs && \
    touch build.rs kernels/cpu_ternary_gemm.cpp kernels/cpu_ternary_gemm_avx2.cpp kernels/fp32_gemm.cpp kernels/memory_paging.cpp kernels/msa_router.cpp

# Download and compile all dependencies
RUN cargo build --release || true

# Now copy the actual source code
COPY src ./src
COPY kernels ./kernels
COPY build.rs ./

# Touch main.rs to invalidate the previous dummy build cache
RUN touch src/main.rs

# Build the release binary (CUDA features can be added via --features cuda later)
RUN cargo build --release

# Stage 2: Create a minimal runtime image using Arch Linux
FROM archlinux:latest

RUN pacman -Syu --noconfirm && pacman -S --noconfirm \
    openssl \
    ca-certificates \
    && rm -rf /var/cache/pacman/pkg/*

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /usr/src/app/target/release/cpu_ai_framework /usr/local/bin/

# Expose standard port for future API
EXPOSE 8080

# Command to run benchmarks/training natively
CMD ["cpu_ai_framework"]
