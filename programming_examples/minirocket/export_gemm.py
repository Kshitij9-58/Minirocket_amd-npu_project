import json
import numpy as np
import sys

# Import minirocket
try:
    import minirocket
except ImportError:
    print("Error: minirocket.py not found.")
    sys.exit(1)

def tile_for_npu(matrix):
    """
    Transform a (512, 512) Linear matrix into Tiled 32x32 Layout.
    Logic: (Rows, Cols) -> (RowTiles, TileH, ColTiles, TileW) -> (RowTiles, ColTiles, TileH, TileW)
    """
    rows, cols = matrix.shape
    tile_h, tile_w = 32, 32
    
    # Reshape to (16, 32, 16, 32)
    reshaped = matrix.reshape(rows // tile_h, tile_h, cols // tile_w, tile_w)
    
    # Transpose to (16, 16, 32, 32) -> Put tiles side-by-side
    tiled = reshaped.transpose(0, 2, 1, 3)
    
    # Flatten back to 1D array
    return tiled.flatten()

def export_data():
    print("--- MiniRocket Data Exporter (Tiled) ---")

    # 1. Load Model
    print("1. Loading 'minirocket_model.json'...")
    try:
        with open('minirocket_model.json', 'r') as f:
            model_data = json.load(f)
        
        weights_raw = np.array(model_data['classifier_coef'])
        if weights_raw.ndim > 1: weights = weights_raw[0]
        else: weights = weights_raw
        
        # Load Parameters
        dilations = np.array(model_data['dilations'], dtype=np.int32)
        num_features_pd = np.array(model_data['num_features_per_dilation'], dtype=np.int32)
        biases = np.array(model_data['biases'], dtype=np.float32)
        parameters = (dilations, num_features_pd, biases)
        
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        return

    # 2. Load Input
    print("2. Loading Input...")
    try:
        with open('minirocket_model_test_data.json', 'r') as f:
            test_json = json.load(f)
        input_data = np.array(test_json['X_test'], dtype=np.float32).reshape(1, -1)
    except Exception as e:
        print(f"   ERROR loading input: {e}")
        return

    # 3. Generate Features
    print("3. Generating Features...")
    X_feat = minirocket.transform(input_data, parameters)
    X_feat_int = (X_feat * 100).astype(np.int16)

    # 4. Prepare Matrices
    LIMIT = 512
    print(f"4. Padding and Scaling (Limit: {LIMIT})...")

    # Matrix A
    Matrix_A = np.zeros((512, 512), dtype=np.int16)
    feat_len = min(X_feat_int.shape[1], LIMIT)
    Matrix_A[0, :feat_len] = X_feat_int[0, :feat_len]

    # Matrix B (Apply Scaling * 1000)
    Matrix_B = np.zeros((512, 512), dtype=np.int32)
    weight_len = min(weights.shape[0], LIMIT)
    scaled_weights = (weights[:weight_len] * 1000).astype(np.int32)
    Matrix_B[:weight_len, 0] = scaled_weights

   # 5. TILE THE DATA (Fixed for NPU Access Order)
    print("5. Tiling data into 32x32 blocks...")
    
    # Matrix A is read Row-wise: Tile normally
    Tiled_A = tile_for_npu(Matrix_A)
    
    # Matrix B is read Column-wise: Transpose BEFORE tiling
    # This ensures Block(1,0) follows Block(0,0) in the file stream
    Tiled_B = tile_for_npu(Matrix_B.T) 

    # 6. Save
    print("6. Saving .txt files...")
    np.savetxt("input_a.txt", Tiled_A, fmt='%d')
    np.savetxt("input_b.txt", Tiled_B, fmt='%d')
    print("Done.")

if __name__ == "__main__":
    export_data()