module {
  aie.device(npu1_1col) {
    func.func private @zero_f32(memref<64x32xf32>)
    func.func private @matmul_bf16_f32(memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xf32>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inA(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memA(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inA] -> [@memA]([] [])
    aie.objectfifo @inB(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @memB(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo.link [@inB] -> [@memB]([] [])
    aie.objectfifo @memC(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>> 
    aie.objectfifo @outC(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x32xf32>> 
    aie.objectfifo.link [@memC] -> [@outC]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<64x32xf32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xf32>> -> memref<64x32xf32>
          func.call @zero_f32(%1) : (memref<64x32xf32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            func.call @matmul_bf16_f32(%3, %5, %1) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xf32>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<262144xbf16>, %arg1: memref<262144xbf16>, %arg2: memref<262144xf32>) {
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 16, 64, 32][32768, 32, 512, 1]) {id = 0 : i64, metadata = @outC} : memref<262144xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 8, 64, 64][0, 64, 512, 1]) {id = 1 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 2 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][16, 8, 64, 64][0, 64, 512, 1]) {id = 3 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 4 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 65536][2, 16, 64, 32][32768, 32, 512, 1]) {id = 8 : i64, metadata = @outC} : memref<262144xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][16, 8, 64, 64][0, 64, 512, 1]) {id = 9 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 10 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][16, 8, 64, 64][0, 64, 512, 1]) {id = 11 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 12 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 131072][2, 16, 64, 32][32768, 32, 512, 1]) {id = 0 : i64, metadata = @outC} : memref<262144xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][16, 8, 64, 64][0, 64, 512, 1]) {id = 1 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 2 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][16, 8, 64, 64][0, 64, 512, 1]) {id = 3 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 4 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 196608][2, 16, 64, 32][32768, 32, 512, 1]) {id = 8 : i64, metadata = @outC} : memref<262144xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][16, 8, 64, 64][0, 64, 512, 1]) {id = 9 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 10 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][16, 8, 64, 64][0, 64, 512, 1]) {id = 11 : i64, metadata = @inA} : memref<262144xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][16, 8, 32, 64][16384, 64, 512, 1]) {id = 12 : i64, metadata = @inB} : memref<262144xbf16>
      aiex.npu.dma_wait {symbol = @outC}
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}

