module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @inA_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x bf16>>
  llvm.mlir.global external @inA_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x bf16>>
  llvm.mlir.global external @memA_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x bf16>>
  llvm.mlir.global external @memA_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<64 x bf16>>
  llvm.mlir.global external @inB_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x bf16>>
  llvm.mlir.global external @inB_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x bf16>>
  llvm.mlir.global external @memB_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x bf16>>
  llvm.mlir.global external @memB_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x bf16>>
  llvm.mlir.global external @memC_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x f32>>
  llvm.mlir.global external @memC_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x f32>>
  llvm.mlir.global external @memC_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x f32>>
  llvm.mlir.global external @memC_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<64 x array<32 x f32>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @zero_f32(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @matmul_bf16_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @memC_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @memA_cons_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @memB_cons_buff_1 : !llvm.ptr
    %3 = llvm.mlir.addressof @memA_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @memB_cons_buff_0 : !llvm.ptr
    %5 = llvm.mlir.addressof @memC_buff_0 : !llvm.ptr
    %6 = llvm.mlir.constant(53 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(48 : i32) : i32
    %9 = llvm.mlir.constant(51 : i32) : i32
    %10 = llvm.mlir.constant(49 : i32) : i32
    %11 = llvm.mlir.constant(52 : i32) : i32
    %12 = llvm.mlir.constant(1 : i32) : i32
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(2 : index) : i64
    %15 = llvm.mlir.constant(8 : index) : i64
    %16 = llvm.mlir.constant(128 : index) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(4294967295 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%17 : i64)
  ^bb1(%20: i64):  // 2 preds: ^bb0, ^bb10
    %21 = llvm.icmp "slt" %20, %18 : i64
    llvm.cond_br %21, ^bb2(%17 : i64), ^bb11
  ^bb2(%22: i64):  // 2 preds: ^bb1, ^bb9
    %23 = llvm.icmp "slt" %22, %16 : i64
    llvm.cond_br %23, ^bb3, ^bb10
  ^bb3:  // pred: ^bb2
    llvm.call @llvm.aie2.acquire(%11, %13) : (i32, i32) -> ()
    %24 = llvm.getelementptr %5[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x f32>>
    llvm.call @zero_f32(%24) : (!llvm.ptr) -> ()
    llvm.br ^bb4(%17 : i64)
  ^bb4(%25: i64):  // 2 preds: ^bb3, ^bb5
    %26 = llvm.icmp "slt" %25, %15 : i64
    llvm.cond_br %26, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %27 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x bf16>>
    %28 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x bf16>>
    llvm.call @matmul_bf16_f32(%28, %27, %24) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %29 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x bf16>>
    %30 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x bf16>>
    llvm.call @matmul_bf16_f32(%30, %29, %24) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %12) : (i32, i32) -> ()
    %31 = llvm.add %25, %14 : i64
    llvm.br ^bb4(%31 : i64)
  ^bb6:  // pred: ^bb4
    llvm.call @llvm.aie2.release(%6, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%11, %13) : (i32, i32) -> ()
    %32 = llvm.getelementptr %0[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x f32>>
    llvm.call @zero_f32(%32) : (!llvm.ptr) -> ()
    llvm.br ^bb7(%17 : i64)
  ^bb7(%33: i64):  // 2 preds: ^bb6, ^bb8
    %34 = llvm.icmp "slt" %33, %15 : i64
    llvm.cond_br %34, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %35 = llvm.getelementptr %4[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x bf16>>
    %36 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x bf16>>
    llvm.call @matmul_bf16_f32(%36, %35, %32) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%10, %13) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%9, %13) : (i32, i32) -> ()
    %37 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<32 x bf16>>
    %38 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<64 x array<64 x bf16>>
    llvm.call @matmul_bf16_f32(%38, %37, %32) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.call @llvm.aie2.release(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%7, %12) : (i32, i32) -> ()
    %39 = llvm.add %33, %14 : i64
    llvm.br ^bb7(%39 : i64)
  ^bb9:  // pred: ^bb7
    llvm.call @llvm.aie2.release(%6, %12) : (i32, i32) -> ()
    %40 = llvm.add %22, %14 : i64
    llvm.br ^bb2(%40 : i64)
  ^bb10:  // pred: ^bb2
    %41 = llvm.add %20, %19 : i64
    llvm.br ^bb1(%41 : i64)
  ^bb11:  // pred: ^bb1
    llvm.return
  }
}
