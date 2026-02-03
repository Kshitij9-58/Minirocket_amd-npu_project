//===- minirocket_runner.cpp ----------------------------------------------===//
// RESTORED: Matches test.cpp Signature (8 Arguments)
// 1. Opcode, 2. Instr, 3. Size, 4. A, 5. B, 6. Out, 7. Tmp, 8. Trace
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATA_TYPE = int32_t;
const int MATRIX_SIZE = 512; 

std::vector<uint32_t> load_instr_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open instruction file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid binary size.");
    std::vector<uint32_t> buffer(size / 4);
    if (!file.read((char*)buffer.data(), size)) throw std::runtime_error("Read failed.");
    return buffer;
}

int main(int argc, char *argv[]) {
    std::string xclbin_path = "build/final_512x512x512_32x32x32.xclbin";
    std::string insts_path = "build/insts_512x512x512_32x32x32.txt";
    std::string kernel_name = "MLIR_AIE";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
    }

    try {
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        auto kernel = xrt::kernel(context, kernel_name);
        
        std::vector<uint32_t> instr_v = load_instr_binary(insts_path);
        int VOLUME = MATRIX_SIZE * MATRIX_SIZE;
        
        // --- 8-ARGUMENT BUFFER SETUP (Matching test.cpp) ---
        // Arg 0: Opcode (Scalar 3) - No BO needed
        // Arg 1: Instructions (Group 1)
        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        
        // Arg 2: Size (Scalar) - No BO needed
        
        // Arg 3: Input A (Group 3)
        auto bo_a = xrt::bo(device, VOLUME * sizeof(DATA_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        
        // Arg 4: Input B (Group 4)
        auto bo_b = xrt::bo(device, VOLUME * sizeof(DATA_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        
        // Arg 5: Output (Group 5)
        auto bo_out = xrt::bo(device, VOLUME * sizeof(DATA_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        // Arg 6: Temp (Group 6) - Dummy buffer
        auto bo_tmp = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

        // Arg 7: Trace (Group 7) - Dummy buffer
        auto bo_trace = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));
        // ---------------------------------------------------

        std::vector<DATA_TYPE> raw_A, raw_B;
        DATA_TYPE val;
        
        std::ifstream fileA("input_a.txt");
        if (!fileA) throw std::runtime_error("Missing input_a.txt");
        while (fileA >> val) raw_A.push_back(val);
        
        std::ifstream fileB("input_b.txt");
        if (!fileB) throw std::runtime_error("Missing input_b.txt");
        while (fileB >> val) raw_B.push_back(val);

        std::cout << "Writing Buffers..." << std::endl;
        bo_instr.write(instr_v.data());
        bo_a.write(raw_A.data()); 
        bo_b.write(raw_B.data());
        
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::cout << "Dispatching Kernel (Opcode 3)..." << std::endl;
        
        // The Magic 8-Argument Call
        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp, bo_trace);
        run.wait();

        std::vector<DATA_TYPE> result(VOLUME);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(result.data());

        std::cout << "Prediction Score: " << result[0] << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
