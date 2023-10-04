from datasets import load_dataset
import tiktoken
import time
import numpy as np
import os

"""Copy these two values from the modelConfig in train.py"""
sequence_length = 1024
encoding_type = 'gpt2'

"""Make every sequence of the same length for batching"""
def pad_sequence(sequence, max_length=sequence_length):
    if len(sequence) < max_length:
        sequence.extend([-1] * (max_length - len(sequence)))
    return sequence[:max_length]

dataset = load_dataset("Bingsu/openwebtext_20p")
encoder = tiktoken.get_encoding(encoding_type)

dataset = dataset['train'][:]
output_dir = 'encoded_data_chunks'
os.makedirs(output_dir, exist_ok=True)

"""Encode, pad, and save the dataset to files in chunks"""
chunk_size = 32768
encoded_chunk = []

chunk_num = 1
t0 = time.time()
for i, text in enumerate(dataset['text']):
    encoded_text = encoder.encode(text) + [50256]
    padded_encoded_text = pad_sequence(encoded_text, sequence_length)
    encoded_chunk.append(padded_encoded_text)
    
    if len(encoded_chunk) >= chunk_size or i == len(dataset) - 1:
        chunk_start = i - len(encoded_chunk) + 1
        chunk_end = i
        chunk_filename = os.path.join(output_dir, f'encoded_chunk_{chunk_start}_{chunk_end}.npy')
        np.save(chunk_filename, np.array(encoded_chunk))
        encoded_chunk = []

        if chunk_num % 50 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Encoded {chunk_num}/{1013} chunks, {dt*1000:.2f}ms")
        
        chunk_num += 1
