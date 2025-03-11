pip install cmake
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON  # For GPU support
make -j