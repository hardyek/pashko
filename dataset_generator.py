import os
import shutil
import random

"""
Copy chunks into dataset_chunks
"""

destination_folder = "dataset_chunks"
os.makedirs(destination_folder, exist_ok=True)

source_folder = "encoded_data_chunks"

npy_files = [f for f in os.listdir(source_folder) if f.endswith(".npy")]

for npy_file in npy_files:
    shutil.copy(os.path.join(source_folder, npy_file), os.path.join(destination_folder, npy_file))

"""
Delete first chunk (it only contains 1 sequence) and shuffle the rest.
"""

file_to_delete = os.path.join(destination_folder, "encoded_chunk_0_0.npy")

if os.path.exists(file_to_delete):
    os.remove(file_to_delete)

remaining_files = [f for f in os.listdir(destination_folder) if f.endswith(".npy") and f != "encoded_chunk_0_0.npy"]

random.shuffle(remaining_files)

for i, filename in enumerate(remaining_files, start=1):
    original_path = os.path.join(destination_folder, filename)
    new_filename = f"chunk_{i}.npy"
    new_path = os.path.join(destination_folder, new_filename)
    os.rename(original_path, new_path)