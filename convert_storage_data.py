"""
将 training_data/ 下所有 storage_data*.mat 转换为 .npz 格式。
保持数组形状不变（(1, N)），确保下游 storage_data['key'][0] 兼容。
"""
import os
import glob
import numpy as np
import scipy.io

DATA_DIR = 'training_data'


def convert_one(mat_path):
    npz_path = mat_path.replace('.mat', '.npz')
    if os.path.exists(npz_path):
        print(f"  SKIP (already exists): {npz_path}")
        return False

    d = scipy.io.loadmat(mat_path)
    # 过滤 MATLAB 元数据 key（以 __ 开头）
    clean = {k: v for k, v in d.items() if not k.startswith('__')}

    np.savez_compressed(npz_path, **clean)
    print(f"  OK: {mat_path} -> {npz_path}")
    return True


def main():
    pattern = os.path.join(DATA_DIR, 'storage_data*.mat')
    mat_files = sorted(glob.glob(pattern))
    print(f"Found {len(mat_files)} storage_data .mat files\n")

    converted = 0
    for mat_path in mat_files:
        if convert_one(mat_path):
            converted += 1

    print(f"\nDone: converted {converted}, skipped {len(mat_files) - converted}")


if __name__ == '__main__':
    main()
