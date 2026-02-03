module @minirocket {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %in_ts = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "in_ts"} : memref<128xf32> 
    %in_bias = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "in_bias"} : memref<84xf32> 
    %out_feat = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "out_feat"} : memref<84xf32> 
    %core_0_2 = aie.core(%tile_0_2) {
      func.call @minirocket_kernel(%in_ts, %in_bias, %out_feat) : (memref<128xf32>, memref<84xf32>, memref<84xf32>) -> ()
      aie.end
    }
    func.func private @minirocket_kernel(memref<128xf32>, memref<84xf32>, memref<84xf32>)
  }
}
