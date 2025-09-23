#!/bin/bash
set -x  # print commands as they run

rm -rf extern/local
rm -rf extern/kokkos-4.7.00
rm -rf extern/kokkos-kernels-4.7.00

ROOT=${PWD}
GCC=$(which gcc)

echo "Root          : ${ROOT}"
echo "Using compiler: ${GCC}"

# ----- pick a reasonable parallelism (leave 2 cores free) -----
calc_jobs() {
  # Try Linux, POSIX, then macOS; fall back to 4
  local cores
  cores=$( nproc 2>/dev/null \
        || getconf _NPROCESSORS_ONLN 2>/dev/null \
        || sysctl -n hw.ncpu 2>/dev/null \
        || echo 4 )
  local j=$(( cores - 2 ))
  if [ "$j" -lt 1 ]; then j=1; fi
  echo "$j"
}

# Allow manual override via MAX_THREADS, else auto-calc
JOBS="${MAX_THREADS:-$(calc_jobs)}"
echo "Building with $JOBS parallel jobs (override with MAX_THREADS)."

# update all submodules
git submodule update --init --recursive

# make local folder for all includes
mkdir -p extern
cd extern && rm -rf local && mkdir local && cd "${ROOT}"

# --- get kokkos-kernels 4.7.00 ---
cd extern && rm -f kokkos-kernels-4.7.00.tar.gz && cd ${ROOT}
cd extern && rm -rf kokkos-kernels-4.7.00/ && cd ${ROOT}
cd extern && \
  wget https://github.com/kokkos/kokkos-kernels/releases/download/4.7.00/kokkos-kernels-4.7.00.tar.gz && \
  tar -xvzf kokkos-kernels-4.7.00.tar.gz && cd ${ROOT}
cd extern && rm -f kokkos-kernels-4.7.00.tar.gz && cd ${ROOT}

# --- get kokkos 4.7.00 ---
cd extern && rm -f kokkos-4.7.00.tar.gz && cd ${ROOT}
cd extern && rm -rf kokkos-4.7.00/ && cd ${ROOT}
cd extern && \
  wget https://github.com/kokkos/kokkos/releases/download/4.7.00/kokkos-4.7.00.tar.gz && \
  tar -xvzf kokkos-4.7.00.tar.gz && cd ${ROOT}
cd extern && rm -f kokkos-4.7.00.tar.gz && cd ${ROOT}

# install prefixes
rm -rf extern/local/kokkos && mkdir -p extern/local/kokkos
rm -rf extern/local/kokkos-kernels && mkdir -p extern/local/kokkos-kernels

# --- build & install kokkos 4.7.00 ---
cd ${ROOT}/extern/kokkos-4.7.00
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${ROOT}/extern/local/kokkos \
  -DKokkos_ARCH_NATIVE=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_TESTS=OFF \
  -DKokkos_ENABLE_EXCEPTIONS=OFF
make install -j "$JOBS"
cd ${ROOT}

# --- build & install kokkos-kernels 4.7.00 (via CMake) ---
cd ${ROOT}/extern/kokkos-kernels-4.7.00
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${ROOT}/extern/local/kokkos-kernels \
  -DCMAKE_PREFIX_PATH=${ROOT}/extern/local/kokkos \
  -DKokkosKernels_ENABLE_TESTS=OFF \
  -DKokkosKernels_ENABLE_EXAMPLES=OFF \
  -DKokkosKernels_ENABLE_PERFTESTS=OFF
make install -j "$JOBS"
cd ${ROOT}

# --- build your app ---
rm -rf build && mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${ROOT}/extern/local/kokkos;${ROOT}/extern/local/kokkos-kernels"
cmake --build . --parallel "$JOBS" --target GPUHeiProMap
