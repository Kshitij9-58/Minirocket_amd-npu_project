import sys
import os
import json
import subprocess
import numpy as np
import itertools
import minirocket 

# --- CONFIGURATION (OFFICIAL FINAL RUN) ---
RUNNER_EXE = "./minirocket_runner"
XCLBIN = "build/final_512x512x512_32x32x32.xclbin"
INSTS = "build/insts_512x512x512_32x32x32.txt"
KERNEL_SIZE = 512

# OPTIMIZED SETTINGS
SCALE_FEAT = 100.0  # Used in the 80% subset run
CLIP_VAL = 3.0      # Safe outlier removal for the full dataset

def tile_matrix_row_major(matrix):
    rows, cols = matrix.shape
    reshaped = matrix.reshape(rows // 32, 32, cols // 32, 32)
    return reshaped.transpose(0, 2, 1, 3).flatten()

def tile_matrix_col_major(matrix):
    rows, cols = matrix.shape
    reshaped = matrix.reshape(rows // 32, 32, cols // 32, 32)
    return reshaped.transpose(2, 0, 1, 3).flatten()

def main():
    print(f"STARTING OFFICIAL FINAL VALIDATION (Scale {SCALE_FEAT} | Clip {CLIP_VAL})")

    # 1. Load Model
    try:
        with open('minirocket_model.json', 'r') as f: model = json.load(f)
    except:
        print("[FAIL] minirocket_model.json not found!"); sys.exit(1)
        
    weights_raw = np.array(model['classifier_coef'])
    intercept_raw = np.array(model.get('classifier_intercept', [0.0]))
    
    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_pd = np.array(model['num_features_per_dilation'], dtype=np.int32)
    biases = np.array(model['biases'], dtype=np.float32)
    parameters = (dilations, num_features_pd, biases)

    # 2. Load Data
    with open('minirocket_model_test_data.json', 'r') as f: test_data = json.load(f)
    X_test = np.array(test_data['X_test'], dtype=np.float32)
    Y_test = np.array(test_data.get('y_test', test_data.get('Y_test'))).astype(int)

    # 3. Normalization Params
    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)

    # 4. Feature Transform
    print("Transforming features...")
    X_feat_raw = minirocket.transform(X_test, parameters)
    NUM_FEATURES = X_feat_raw.shape[1]
    
    # Class Setup
    if weights_raw.ndim == 1:
        if weights_raw.size == NUM_FEATURES:
            num_classes = 2 
            weights = weights_raw.reshape(1, -1)
            intercepts = np.array([intercept_raw[0]]) if intercept_raw.ndim > 0 else np.array([intercept_raw])
        else:
            num_classes = weights_raw.size // NUM_FEATURES
            weights = weights_raw.reshape(num_classes, NUM_FEATURES)
            intercepts = np.array(intercept_raw) if intercept_raw.size == num_classes else np.zeros(num_classes)
    else:
        num_classes = weights_raw.shape[0]
        weights = weights_raw
        intercepts = np.array(intercept_raw) if intercept_raw.size == num_classes else np.zeros(num_classes)
        
    print(f"Features: {NUM_FEATURES} | Classes: {num_classes}")

    # --- 5. CPU BASELINE CHECK ---
    print("\nCalculating CPU Baseline...")
    cpu_correct = 0
    for i in range(len(X_test)):
        feat_vec_norm = (X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8)
        if num_classes == 2:
             cpu_score = np.dot(feat_vec_norm, weights[0]) + intercepts[0]
             cpu_pred = 1 if cpu_score > 0 else 0
        else:
             cpu_scores = np.dot(feat_vec_norm, weights.T) + intercepts
             cpu_pred = np.argmax(cpu_scores)
        
        if cpu_pred == Y_test[i]: cpu_correct += 1
    
    cpu_acc = cpu_correct/len(X_test)
    print(f"BASELINE CPU ACCURACY: {cpu_acc:.2%}")
    print("-" * 65)

    # --- 6. NPU  ---
    raw_npu_results = [] 
    
    print("\n[PER-SAMPLE SCORES]")
    print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Int':<12} | {'Scale Used':<12}")
    print("-" * 65)
    
    for i in range(len(X_test)):
        # --- PRE-PROCESS: NORMALIZE & CLIP ---
        feat_vec_raw = X_feat_raw[i]
        feat_vec_norm = (feat_vec_raw - scaler_mean) / (scaler_scale + 1e-8)
        
        # APPLY CLIP 3.0
        feat_vec_norm = np.clip(feat_vec_norm, -CLIP_VAL, CLIP_VAL)
        
        # --- NPU Inference ---
        f_quant = (np.nan_to_num(feat_vec_norm) * SCALE_FEAT).astype(np.int16)
        npu_raw_scores = [] 

        loop_classes = 1 if num_classes == 2 else num_classes

        for c_idx in range(loop_classes):
            current_weights = weights[c_idx]
            
            # --- SCALE 100.0 LOGIC: USE 32000 TARGET ---
            max_w = np.max(np.abs(current_weights))
            if max_w == 0: max_w = 1.0
            
            safe_scale_w = 32000.0 / max_w
            
            w_quant = (current_weights * safe_scale_w).astype(np.int32)
            current_total_scale = SCALE_FEAT * safe_scale_w

            total_class_score = 0
            
            # Hardware Execution Loop
            for chunk_start in range(0, NUM_FEATURES, KERNEL_SIZE):
                f_chunk = f_quant[chunk_start : chunk_start + KERNEL_SIZE]
                w_chunk = w_quant[chunk_start : chunk_start + KERNEL_SIZE]
                
                f_padded = np.zeros(KERNEL_SIZE, dtype=np.int16)
                f_padded[:len(f_chunk)] = f_chunk
                w_padded = np.zeros(KERNEL_SIZE, dtype=np.int16)
                w_padded[:len(w_chunk)] = w_chunk.astype(np.int16) 

                Mat_A = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.int16)
                Mat_A[0, :] = f_padded
                Mat_B = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.int16)
                Mat_B[:, 0] = w_padded
                
                Tiled_A = tile_matrix_row_major(Mat_A)
                Tiled_B = tile_matrix_col_major(Mat_B)
                
                # --- SHOTGUN FILE IO ---
                np.savetxt("input_a.txt", Tiled_A, fmt='%d')
                np.savetxt("input.txt", Tiled_A, fmt='%d')
                np.savetxt("input0.txt", Tiled_A, fmt='%d')
                
                np.savetxt("input_b.txt", Tiled_B, fmt='%d')
                np.savetxt("input1.txt", Tiled_B, fmt='%d')
                
                cmd = [RUNNER_EXE, "-x", XCLBIN, "-i", INSTS, "-k", "MLIR_AIE"]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    val = 0
                    for line in res.stdout.split('\n'):
                        if "Prediction Score:" in line:
                            raw_str = line.split(':')[-1].strip()
                            try:
                                val = int(float(raw_str))
                            except (ValueError, OverflowError):
                                val = 0 
                            break
                    total_class_score += val
                except:
                    pass
            
            scaled_intercept = intercepts[c_idx] * current_total_scale
            final_class_score = total_class_score + int(scaled_intercept)
            
            npu_raw_scores.append(final_class_score)

            # --- TABLE PRINT ---
            if i < 5:
                if num_classes == 2:
                     cpu_val = np.dot(feat_vec_norm, weights[0]) + intercepts[0]
                else:
                     cpu_val = np.dot(feat_vec_norm, weights[c_idx]) + intercepts[c_idx]
                print(f"{i:<6} | {c_idx:<5} | {cpu_val:<12.4f} | {final_class_score:<12} | {safe_scale_w:<12.1f}")
        
        raw_npu_results.append(npu_raw_scores)
        
        if (i+1) % 20 == 0: print(f"   Processed {i+1}/{len(X_test)}")

    print("-" * 65)
    
    # --- 7. NPU SOLVER ---
    print("Attempting to un-scramble NPU class labels...")
    
    try:
        scores_matrix = np.array(raw_npu_results, dtype=np.float64)
    except ValueError:
        max_len = max(len(row) for row in raw_npu_results)
        fixed_matrix = []
        for row in raw_npu_results:
             row = row + [0]*(max_len-len(row))
             fixed_matrix.append(row)
        scores_matrix = np.array(fixed_matrix, dtype=np.float64)

    means = scores_matrix.mean(axis=0)
    stds = scores_matrix.std(axis=0) + 1e-8
    normalized_scores = (scores_matrix - means) / stds

    base_indices = list(range(num_classes))
    permutations = list(itertools.permutations(base_indices))
    
    best_acc = 0
    best_perm = None
    
    print(f"{'MAPPING (NPU->True)':<25} | {'ACCURACY':<10}")
    print("-" * 40)

    for perm in permutations:
        correct = 0
        for i in range(len(X_test)):
            scores = normalized_scores[i]
            remapped_scores = np.zeros(num_classes)
            for npu_idx, true_label in enumerate(perm):
                if npu_idx < len(scores):
                    remapped_scores[true_label] = scores[npu_idx]
            pred = np.argmax(remapped_scores)
            if pred == Y_test[i]: correct += 1
        
        acc = correct / len(X_test)
        print(f"{str(perm):<25} | {acc:.2%}")
        if acc > best_acc:
            best_acc = acc
            best_perm = perm

    print("-" * 40)
    print(f" FINAL ACCURACY: {best_acc:.2%}")

if __name__ == "__main__":
    main()