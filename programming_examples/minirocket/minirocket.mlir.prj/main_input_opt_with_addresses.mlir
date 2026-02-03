module @minirocket attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @out_feat() {addr_space = 0 : i32} : !llvm.array<84 x f32>
  llvm.mlir.global external @in_bias() {addr_space = 0 : i32} : !llvm.array<84 x f32>
  llvm.mlir.global external @in_ts() {addr_space = 0 : i32} : !llvm.array<128 x f32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @minirocket_kernel(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @out_feat : !llvm.ptr
    %1 = llvm.mlir.addressof @in_bias : !llvm.ptr
    %2 = llvm.mlir.addressof @in_ts : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x f32>
    %4 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<84 x f32>
    %5 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<84 x f32>
    llvm.call @minirocket_kernel(%3, %4, %5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}
