module @minirocket {
  // WRAPPER: define the target device (Ryzen AI NPU)
  aie.device(npu1_1col) {

    // Define Tile (Column 0, Row 2)
    %tile02 = aie.tile(0, 2)
    
    // Define Memory Buffers (Object FIFOs)
    // 128 input floats, 84 bias floats, 84 output floats
    %in_ts = aie.buffer(%tile02) { sym_name = "in_ts" } : memref<128xf32>
    %in_bias = aie.buffer(%tile02) { sym_name = "in_bias" } : memref<84xf32>
    %out_feat = aie.buffer(%tile02) { sym_name = "out_feat" } : memref<84xf32>

    // Define the Core Logic (Processor)
    %core02 = aie.core(%tile02) {
      // Call the external C++ function defined in kernel.cc
      func.call @minirocket_kernel(%in_ts, %in_bias, %out_feat) : (memref<128xf32>, memref<84xf32>, memref<84xf32>) -> ()
      aie.end
    }
    
    // Declaration of the external C++ kernel function signature
    func.func private @minirocket_kernel(memref<128xf32>, memref<84xf32>, memref<84xf32>) -> ()

  }
}