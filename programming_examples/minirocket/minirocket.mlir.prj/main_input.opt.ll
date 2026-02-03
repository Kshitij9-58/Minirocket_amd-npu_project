; ModuleID = '/home/wch464/mlir-aie/programming_examples/minirocket/minirocket.mlir.prj/main_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@out_feat = external global [84 x float]
@in_bias = external global [84 x float]
@in_ts = external global [128 x float]

declare void @minirocket_kernel(ptr, ptr, ptr) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  tail call void @minirocket_kernel(ptr nonnull @in_ts, ptr nonnull @in_bias, ptr nonnull @out_feat)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
