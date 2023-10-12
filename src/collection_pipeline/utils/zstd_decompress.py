import io
import os
import json
import base64 
import subprocess
import zstandard as zstd

from pathlib import Path
from time import sleep, time
from os.path import isfile, join as pjoin
from paths import COMPRESSED_SUMBISSIONS, DECOMPRESSED_SUMBISSIONS


def decompress_zstd(input_directory, output_directory):
    """
    Decompresses all files in input_directory and saves them in output_directory
    """
    files = os.listdir(input_directory)

    compressed_files = os.listdir(input_directory)
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    
    for file in compressed_files: 
        if file.__contains__('2018'):
            file_rb = open(pjoin(input_directory, file), 'rb')        
            stream_reader = dctx.stream_reader(file_rb)
            f = io.TextIOWrapper(stream_reader, encoding='utf-8')
            
            json_array = []
            for index, l in enumerate(f): 
                json_array.append(json.loads(l))

            write_data(pjoin(output_directory, f"{file.split('.')[0]}.json"), json_array)

def load_data(file):
    """
    Returns a json object loaded from a file. 
    """
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f) 
    return (data)

def write_data(file, data):
    """
    Writes json object to a file.
    """
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def merge_json(json_array):
    items = {}
    for item in json_array:
        items = items | item
        print(items)

    return {"items": items}


def main(): 
    start_time = time()

    print("Started decompression process at: ", start_time)    
    
    decompress_zstd(input_directory=COMPRESSED_SUMBISSIONS, output_directory=DECOMPRESSED_SUMBISSIONS)
    
    print("Finished decompression process at:",  time())
    

if __name__ == "__main__": 
    main()
    
