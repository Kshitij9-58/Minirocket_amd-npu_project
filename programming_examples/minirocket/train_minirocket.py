#!/usr/bin/env python3

import numpy as np
import json
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
import os
from itertools import combinations

# sktime datasets import will be done locally in functions

class MiniRocket:
    """MiniRocket implementation matching sktime reference implementation"""

    def __init__(self, num_kernels=10000, max_dilations_per_kernel=32, random_state=None):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.kernels = None
        self.dilations = None
        self.num_features_per_dilation = None
        self.biases = None
        self.parameters = None
        
    def _quantiles(self, n):
        return np.array(
            [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
        )

    def _fit_dilations(self, n_timepoints, num_features, max_dilations_per_kernel):
        num_kernels = 84
        if num_features < 84:
            num_features = 84

        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    def _fit_biases(self, X, dilations, num_features_per_dilation, quantiles, seed):
        if seed is not None:
            np.random.seed(seed)

        n_instances, n_timepoints = X.shape
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
        num_kernels = len(indices)
        num_dilations = len(dilations)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        biases = np.zeros(num_features, dtype=np.float32)
        feature_index_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation
                _X = X[np.random.randint(n_instances)]
                A = -_X
                G = _X + _X + _X
                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G
                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
                biases[feature_index_start:feature_index_end] = np.quantile(
                    C, quantiles[feature_index_start:feature_index_end]
                )
                feature_index_start = feature_index_end

        return biases

    def fit(self, X):
        X = X.astype(np.float32)
        _, n_timepoints = X.shape
        self.time_series_length = n_timepoints

        if n_timepoints < 9:
            raise ValueError(f"n_timepoints must be >= 9, but found {n_timepoints}")

        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        seed = np.int32(self.random_state) if isinstance(self.random_state, int) else None

        dilations, num_features_per_dilation = self._fit_dilations(
            n_timepoints, self.num_kernels_, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)
        quantiles = self._quantiles(84 * num_features_per_kernel)
        biases = self._fit_biases(X, dilations, num_features_per_dilation, quantiles, seed)

        self.parameters = (dilations, num_features_per_dilation, biases)
        self.dilations = dilations
        self.num_features_per_dilation = num_features_per_dilation
        self.biases = biases
        return self

    def transform(self, X):
        X = X.astype(np.float32)
        n_instances, n_timepoints = X.shape
        dilations, num_features_per_dilation, biases = self.parameters
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
        num_kernels = len(indices)
        num_dilations = len(dilations)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        features = np.zeros((n_instances, num_features), dtype=np.float32)

        for example_index in range(n_instances):
            _X = X[example_index]
            A = -_X
            G = _X + _X + _X
            feature_index_start = 0

            for dilation_index in range(num_dilations):
                _padding0 = dilation_index % 2
                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) // 2
                num_features_this_dilation = num_features_per_dilation[dilation_index]

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G
                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                for kernel_index in range(num_kernels):
                    feature_index_end = feature_index_start + num_features_this_dilation
                    _padding1 = (_padding0 + kernel_index) % 2
                    index_0, index_1, index_2 = indices[kernel_index]
                    C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                    if _padding1 == 0:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(C > biases[feature_index_start + feature_count])
                            )
                    else:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(
                                    C[padding:-padding] > biases[feature_index_start + feature_count]
                                )
                            )
                    feature_index_start = feature_index_end
        return features

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def load_real_dataset(dataset_name='arrow_head'):
    try:
        from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand
        if dataset_name == 'arrow_head':
            X, y = load_arrow_head()
        elif dataset_name == 'gun_point':
            X, y = load_gunpoint()
        elif dataset_name == 'italy_power':
            X, y = load_italy_power_demand()
        else:
            X, y = load_arrow_head()
        
        X_numpy = np.array([series.values for series in X.iloc[:, 0]])
        unique_labels = sorted(set(y))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_numpy = np.array([label_to_int[label] for label in y])
        return X_numpy.astype(np.float32), y_numpy
    except ImportError:
        print("sktime import error, using synthetic")
        return generate_sample_data()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return generate_sample_data()

def generate_sample_data(n_samples=1000, length=128, n_classes=4):
    np.random.seed(42)
    X = np.zeros((n_samples, length))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        class_id = i % n_classes
        if class_id == 0: X[i] = np.sin(np.linspace(0, 4*np.pi, length)) + 0.1 * np.random.randn(length)
        elif class_id == 1: X[i] = np.sign(np.sin(np.linspace(0, 4*np.pi, length))) + 0.1 * np.random.randn(length)
        elif class_id == 2: X[i] = np.cumsum(0.1 * np.random.randn(length))
        else: X[i] = np.exp(-np.linspace(0, 3, length)) + 0.1 * np.random.randn(length)
        y[i] = class_id
    return X.astype(np.float32), y

def save_model_parameters(minirocket, scaler, classifier, filename="minirocket_model.json"):
    kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
    model_data = {
        "num_kernels": 84,
        "num_dilations": len(minirocket.dilations),
        "num_features": len(minirocket.biases),
        "num_classes": len(classifier.classes_),
        "time_series_length": minirocket.time_series_length,
        "kernel_indices": kernel_indices.tolist(),
        "dilations": minirocket.dilations.tolist(),
        "num_features_per_dilation": minirocket.num_features_per_dilation.tolist(),
        "biases": minirocket.biases.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"Model parameters saved to {filename}")
    return model_data

def main():
    parser = argparse.ArgumentParser(description='Train MiniRocket model')
    parser.add_argument('--dataset', type=str, default='arrow_head')
    parser.add_argument('--output', type=str, default='minirocket_model.json')
    args = parser.parse_args()
    
    if args.dataset == 'synthetic':
        X, y = generate_sample_data()
    else:
        print(f"Loading real dataset: {args.dataset}")
        X, y = load_real_dataset(args.dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # --- CHANGED: REMOVED STREAMING LOGIC, USING FULL BATCH ---
    # We want to test on the entire Test Set, not just one streaming example.
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("Training MiniRocket...")
    minirocket = MiniRocket(num_kernels=840, random_state=42)
    X_train_features = minirocket.fit_transform(X_train)
    X_test_features = minirocket.transform(X_test)  # Transform ALL test samples
    
    print(f"Feature shape: {X_train_features.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train classifier
    print("Training classifier...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)
    
    # Test accuracy
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"PYTHON BASELINE ACCURACY: {accuracy:.4f}")
    
    # Save model
    save_model_parameters(minirocket, scaler, classifier, args.output)
    
    # Save FULL test data
    test_data = {
        "dataset_name": args.dataset,
        "X_test": X_test.tolist(),  # Save ALL samples
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_accuracy": float(accuracy),
        "num_samples": len(X_test),
        "series_length": X_test.shape[1],
        "num_classes": len(np.unique(y))
    }
    
    test_filename = args.output.replace('.json', '_test_data.json')
    with open(test_filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Full test dataset saved to {test_filename}")

if __name__ == "__main__":
    main()