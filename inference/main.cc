#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;

// Allocate a CPU Tensor
ffi::Tensor AllocTensor(std::vector<int64_t> shape, DLDataType dtype) {
    struct CPUAllocator {
        void AllocData(DLTensor* tensor) {
            size_t size = 1;
            for (int i = 0; i < tensor->ndim; i++) size *= tensor->shape[i];
            tensor->data = std::malloc(size * (tensor->dtype.bits / 8));
        }
        void FreeData(DLTensor* tensor) { std::free(tensor->data); }
    };
    DLDevice cpu = DLDevice({kDLCPU, 0});
    return ffi::Tensor::FromNDAlloc(CPUAllocator(), ffi::ShapeView(shape.data(), shape.size()), dtype, cpu);
}

static bool ReadFile(const std::string& path, void* dst, size_t nbytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(static_cast<char*>(dst), nbytes);
    return static_cast<size_t>(f.gcount()) == nbytes;
}

static bool WriteFile(const std::string& path, const void* src, size_t nbytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(static_cast<const char*>(src), nbytes);
    return static_cast<bool>(f);
}

static void Usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <model.so> [--input in.bin] [--output out.bin] [--runs N]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        Usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string input_path, output_path;
    int runs = 5;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
        else { Usage(argv[0]); return 1; }
    }

    try {
        // Load the model shared library
        ffi::Module mod = ffi::Module::LoadFromFile(model_path);
        std::cout << "Model loaded: " << model_path << std::endl;

        // Load the virtual machine executable from the model
        ffi::Function vm_load = mod->GetFunction("vm_load_executable").value();
        ffi::Module vm_mod = vm_load().cast<ffi::Module>();

        DLDevice dev{kDLCPU, 0};
        ffi::Function vm_init = vm_mod->GetFunction("vm_initialization").value();
        vm_init(static_cast<int>(dev.device_type), dev.device_id,
                static_cast<int>(2));  // 2 = POOLED_ALLOCATOR

        ffi::Function run = vm_mod->GetFunction("main").value();
        std::cout << "VirtualMachine created" << std::endl;

        // Prepare input (1, 3, 640, 640) float32
        DLDataType f32 = DLDataType({kDLFloat, 32, 1});
        ffi::Tensor input = AllocTensor({1, 3, 640, 640}, f32);
        const size_t in_bytes = 1 * 3 * 640 * 640 * sizeof(float);
        if (input_path.empty()) {
            std::memset(input.data_ptr(), 0, in_bytes);
            std::cout << "Input: zeros" << std::endl;
        } else {
            if (!ReadFile(input_path, input.data_ptr(), in_bytes)) {
                std::cerr << "Failed to read " << in_bytes << " bytes from " << input_path
                          << std::endl;
                return 1;
            }
            std::cout << "Input: " << input_path << std::endl;
        }

        // Warm-up (not timed)
        std::cout << "Warm-up run..." << std::endl;
        ffi::Any result = run(input);

        // Timed runs
        std::vector<double> times;
        for (int r = 0; r < runs; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            result = run(input);
            auto end = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            std::cout << "Run " << r + 1 << ": " << times.back() << " ms" << std::endl;
        }
        std::sort(times.begin(), times.end());
        std::cout << "Inference time: median " << times[times.size() / 2] << " ms (min "
                  << times.front() << ", max " << times.back() << ", runs " << runs << ")"
                  << std::endl;

        // Write output: first element of the returned tuple, (1, 84, 8400) float32
        if (!output_path.empty()) {
            ffi::Array<ffi::Any> outs = result.cast<ffi::Array<ffi::Any>>();
            ffi::Tensor out = outs[0].cast<ffi::Tensor>();
            const DLTensor* dl = out.GetDLTensorPtr();
            size_t n = 1;
            for (int i = 0; i < dl->ndim; i++) n *= dl->shape[i];
            if (!WriteFile(output_path, dl->data, n * sizeof(float))) {
                std::cerr << "Failed to write " << output_path << std::endl;
                return 1;
            }
            std::cout << "Output written: " << output_path << " (" << n << " floats)"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}