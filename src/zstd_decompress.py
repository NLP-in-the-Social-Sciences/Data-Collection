import os 
import subprocess


def decompress_zstd(input_directory, output_directory):
    """
    Decompresses all files in input_directory and saves them in output_directory
    """
    files = os.listdir(input_directory)

    for file in files:
        ... 
    
