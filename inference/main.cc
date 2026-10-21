#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;

// Allocate a CPU Tensor
ffi::Tensor AllocTensor(std::vector<int64_t> shape, DLDataType dtype) {
    struct CPUAllocator {
        size_t nbytes;
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.so>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    // Load compiled model
    ffi::Module mod = ffi::Module::LoadFromFile(model_path);
    std::cout << "Model loaded: " << model_path << std::endl;

    // Get VirtualMachine
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
    memset(input.data_ptr(), 0, 1 * 3 * 640 * 640 * sizeof(float));
    std::cout << "Input prepared: (1, 3, 640, 640) float32" << std::endl;

    // Run inference
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    ffi::Any result = run(input);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Inference time: " << ms << " ms" << std::endl;

    return 0;
}