# --- PATHS ---
PEANO_ROOT = /home/wch464/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie
CLANG_CPP  = $(PEANO_ROOT)/bin/clang++
AIECC      = aiecc.py

# --- FLAGS ---
# Use C++20 and the correct AIE2 target
KERNEL_CFLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG \
                -I/home/wch464/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include

# --- TARGETS ---
all: build/final.xclbin

# 1. Compile Kernel to LOCAL directory (kernel.o)
#    We strip the path to ensure aiecc.py picks it up easily
kernel.o: kernel.cc
	@echo "Compiling Kernel..."
	$(CLANG_CPP) $(KERNEL_CFLAGS) -c kernel.cc -o kernel.o
	@echo "Verifying Symbol..."
	@nm kernel.o | grep minirocket_kernel || echo "ERROR: Symbol mangled or missing!"

# 2. Link XCLBIN
#     pass 'kernel.o' (local file) directly
build/final.xclbin: minirocket.mlir kernel.o
	@mkdir -p build
	@echo "Linking NPU Binary..."
	$(AIECC) --aie-generate-cdo --aie-generate-npu --no-compile-host \
		--xclbin-name=$@ --npu-insts-name=build/insts.txt \
		--no-xchesscc \
		--no-xbridge \
		--peano=$(PEANO_ROOT) \
		minirocket.mlir kernel.o

# 3. Run
run: all
	python3 run_inference_npu.py

clean:
	rm -rf build *.exe *.elf *.o kernel.o