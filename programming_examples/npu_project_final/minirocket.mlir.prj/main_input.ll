; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@inA_cons_buff_1 = external global [64 x [64 x bfloat]]
@inA_cons_buff_0 = external global [64 x [64 x bfloat]]
@memA_cons_buff_1 = external global [64 x [64 x bfloat]]
@memA_cons_buff_0 = external global [64 x [64 x bfloat]]
@inB_cons_buff_1 = external global [64 x [32 x bfloat]]
@inB_cons_buff_0 = external global [64 x [32 x bfloat]]
@memB_cons_buff_1 = external global [64 x [32 x bfloat]]
@memB_cons_buff_0 = external global [64 x [32 x bfloat]]
@memC_buff_1 = external global [64 x [32 x float]]
@memC_buff_0 = external global [64 x [32 x float]]
@memC_cons_buff_1 = external global [64 x [32 x float]]
@memC_cons_buff_0 = external global [64 x [32 x float]]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie2.event(i32)

; Unknown intrinsic
declare void @llvm.aie2.put.ms(i32, i32)

; Unknown intrinsic
declare { i32, i32 } @llvm.aie2.get.ss()

; Unknown intrinsic
declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

; Unknown intrinsic
declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

; Unknown intrinsic
declare void @llvm.aie2.acquire(i32, i32)

; Unknown intrinsic
declare void @llvm.aie2.release(i32, i32)

declare void @zero_f32(ptr)

declare void @matmul_bf16_f32(ptr, ptr, ptr)

define void @core_0_2() {
  br label %1

1:                                                ; preds = %21, %0
  %2 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %23

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %20, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 128
  br i1 %6, label %7, label %21

7:                                                ; preds = %4
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @zero_f32(ptr @memC_buff_0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %12, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 8
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @matmul_bf16_f32(ptr @memA_cons_buff_0, ptr @memB_cons_buff_0, ptr @memC_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @matmul_bf16_f32(ptr @memA_cons_buff_1, ptr @memB_cons_buff_1, ptr @memC_buff_0)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %12 = add i64 %9, 2
  br label %8

13:                                               ; preds = %8
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @zero_f32(ptr @memC_buff_1)
  br label %14

14:                                               ; preds = %17, %13
  %15 = phi i64 [ %18, %17 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 8
  br i1 %16, label %17, label %19

17:                                               ; preds = %14
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @matmul_bf16_f32(ptr @memA_cons_buff_0, ptr @memB_cons_buff_0, ptr @memC_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @matmul_bf16_f32(ptr @memA_cons_buff_1, ptr @memB_cons_buff_1, ptr @memC_buff_1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %18 = add i64 %15, 2
  br label %14

19:                                               ; preds = %14
  call void @llvm.aie2.release(i32 53, i32 1)
  %20 = add i64 %5, 2
  br label %4

21:                                               ; preds = %4
  %22 = add i64 %2, 1
  br label %1

23:                                               ; preds = %1
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
