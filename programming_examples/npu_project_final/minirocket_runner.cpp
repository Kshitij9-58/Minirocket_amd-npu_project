//===- minirocket_runner.cpp ----------------------------------------------===//
// UPDATED: Integer Configuration (Scale 80.0 / 100.0 Compatible)
// Input:  int16_t (Standard 16-bit Integer)
// Output: int32_t (Standard 32-bit Integer Accumulator)
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

// --- INTEGER NPU DATATYPES (Matches Current Hardware) ---
// We use int16_t for scaled weights/features
using INPUT_TYPE = int16_t;
// The NPU accumulates into 32-bit integers
using OUTPUT_TYPE = int32_t;

const int MATRIX_SIZE = 512; 
const int VOLUME = MATRIX_SIZE * MATRIX_SIZE;

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
    // Default paths
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
        
        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        
        // --- BUFFER SETUP (INTEGER) ---
        // Inputs are 16-bit Integers
        auto bo_a = xrt::bo(device, VOLUME * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_b = xrt::bo(device, VOLUME * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        
        // Output is 32-bit Integer
        auto bo_out = xrt::bo(device, VOLUME * sizeof(OUTPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        auto bo_tmp = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_trace = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        // --- READ DATA AS INTEGERS ---
        std::vector<INPUT_TYPE> raw_A, raw_B;
        int val_int; // Read as standard int
        
        // Reading the "Shotgun" files
        std::ifstream fileA("input_a.txt");
        if (!fileA) throw std::runtime_error("Missing input_a.txt");
        while (fileA >> val_int) raw_A.push_back(static_cast<INPUT_TYPE>(val_int));
        
        std::ifstream fileB("input_b.txt");
        if (!fileB) throw std::runtime_error("Missing input_b.txt");
        while (fileB >> val_int) raw_B.push_back(static_cast<INPUT_TYPE>(val_int));

        if(raw_A.size() < VOLUME) raw_A.resize(VOLUME, 0);
        if(raw_B.size() < VOLUME) raw_B.resize(VOLUME, 0);

        bo_instr.write(instr_v.data());
        bo_a.write(raw_A.data()); 
        bo_b.write(raw_B.data());
        
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out, bo_tmp, bo_trace);
        run.wait();

        // --- READ BACK AS INTEGER ---
        std::vector<OUTPUT_TYPE> result(VOLUME);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(result.data());

        // Output Integer Score
        std::cout << "Prediction Score: " << result[0] << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}