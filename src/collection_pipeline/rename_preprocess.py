# import umap
import json
import os
import numpy as np
import pandas as pd 
import altair as alt

from paths import *
from tqdm import tqdm
from annoy import AnnoyIndex
from multiprocessing import Pool
from os.path import join as pjoin
from Archive.json_load_write import load_json_data
from multiprocessing import Pool, cpu_count
from src.collection_pipeline.utils.preprocessing import lemmatize
# from ..notebooks.functions.preprocessing import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

def load_filter_dump(file_name):
    if not os.path.isdir(pjoin(DECOMPRESSED_SUMBISSIONS, file_name)): 
         
        input_file_path = pjoin(DECOMPRESSED_SUMBISSIONS, file_name)
        output_file_path = pjoin(DECOMPRESSED_SUMBISSIONS,"csv", file_name.split('.')[0] + ".csv")

        if (not os.path.exists(output_file_path)):
            
                print(f"Processing file {file_name}.")

                chunk_size = 100000
                df = pd.DataFrame()
                df_chunk = pd.read_json(input_file_path, lines=True, chunksize=chunk_size)
                
                for index, chunk in enumerate(df_chunk):
                    chunk = chunk[["id", "selftext", "title", "subreddit", "permalink", "url"]]
                    chunk = chunk.dropna()
                    chunk = chunk.drop_duplicates(subset=["id"])
                    chunk = chunk[chunk['selftext'].str.len() > 150]
                    chunk["lemmatized_selftext"] = chunk["selftext"].apply(lemmatize)
                    
                    df = pd.concat([df, chunk])

                if not df.empty: 
                    df.to_csv(output_file_path, index=False)
                
                print(f"File {file_name} processed.")
        else: 
            print(f"File {file_name.split('.')[0]}.csv already exists.")


def main(): 
    num_cores = 8

    file_list = os.listdir(DECOMPRESSED_SUMBISSIONS)

    print("Starting preprocessing of files.")
    
    # with Pool(num_cores) as p:
    #     p.map(load_filter_dump, file_list)

    for file in file_list:
        load_filter_dump(file)

    print(f"Finished preprocessing {len(file_list)} files.")

if __name__ == '__main__':

    main()