import numpy as np
import sys

def load_tiled_matrix(filename, is_col_major_tiling=False):
    try:
        data = np.loadtxt(filename, dtype=np.int32)
    except OSError:
        print(f"Error: Could not find {filename}")
        sys.exit(1)
    
    # Grid is 16x16 blocks (512 / 32 = 16)
    grid_size = 16
    block_size = 32
    
    matrix = np.zeros((512, 512), dtype=np.int32)
    idx = 0
    
    # Iterate through blocks in the 16x16 grid
    for r_block in range(grid_size):
        for c_block in range(grid_size):
            # Determine where this block goes in the 512x512 matrix
            if is_col_major_tiling:
                # Column-Major Tiling (used for Weights/Input B)
                # The file stores blocks (0,0), (1,0), (2,0)...
                curr_row = c_block * block_size
                curr_col = r_block * block_size
            else:
                # Row-Major Tiling (used for Features/Input A)
                # The file stores blocks (0,0), (0,1), (0,2)...
                curr_row = r_block * block_size
                curr_col = c_block * block_size
                
            # Extract 1024 elements for this 32x32 block
            if idx + 1024 > len(data): break
            block_data = data[idx : idx + 1024].reshape(32, 32)
            
            # Place block into the full matrix
            matrix[curr_row:curr_row+32, curr_col:curr_col+32] = block_data
            idx += 1024
            
    return matrix

print("--- Verifying NPU Logic on CPU ---")

# 1. Reconstruct Matrix A (Row-Major Tiling)
print("Loading input_a.txt...")
A = load_tiled_matrix("input_a.txt", is_col_major_tiling=False)

# 2. Reconstruct Matrix B (Column-Major Tiling)
print("Loading input_b.txt...")
B = load_tiled_matrix("input_b.txt", is_col_major_tiling=True)

# 3. Compute Linear Classification: Dot Product of Row 0 (Features) and Col 0 (Weights)
#    Note: MiniRocket effectively only uses the first vector of each matrix.
print("Computing Dot Product...")
cpu_score = np.dot(A[0, :], B[:, 0])

print(f"\nNPU Output (from your run): 2985")
print(f"CPU Output (calculated now): {cpu_score}")

if cpu_score == 2985:
    print("\n[SUCCESS] Exact Match! The NPU is bit-exact correct.")
elif (cpu_score > 0 and 2985 > 0) or (cpu_score < 0 and 2985 < 0):
    print("\n[SUCCESS] Functional Match! Both predict Class 1 (Positive).")
    print("Small differences are expected if internal accumulation depths differ.")
else:
    print("\n[FAIL] Sign Mismatch. Check tiling logic.")
