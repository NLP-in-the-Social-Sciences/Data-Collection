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
from json_load_write import load_json_data
from multiprocessing import Pool, cpu_count
from preprocessing import lemmatize
# from ..notebooks.functions.preprocessing import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

def load_filter_dump(file_name):

    if not os.path.exists(pjoin(DECOMPRESSED_SUMBISSIONS,"csv", file_name.split('.')[0] + ".csv")):
        print(f"Processing file {file_name}.")
        chunk_size = 100000
        df = pd.DataFrame()
        df_chunk = pd.read_json(os.path.join(DECOMPRESSED_SUMBISSIONS, file_name), lines=True, chunksize=chunk_size)
        
        for index, chunk in enumerate(df_chunk):
            chunk = chunk[["id", "selftext", "title", "subreddit", "permalink", "url"]]
            chunk = chunk.dropna()
            chunk = chunk.drop_duplicates(subset=["id"])
            chunk = chunk[chunk['selftext'].str.len() > 100]
            chunk["lemmatized_selftext"] = chunk["selftext"].apply(lemmatize)
            
            df = pd.concat([df, chunk])
        
        csv_file_name = os.path.join(DECOMPRESSED_SUMBISSIONS, file_name.split('.')[0] + ".csv")
        df.to_csv(csv_file_name, index=False)
        
        print(f"File {file_name} processed.")
    else: 
        print(f"File {file_name.split('.')[0]}.csv already exists.")


def main(): 
    num_cores = 8

    file_list = os.listdir(DECOMPRESSED_SUMBISSIONS)

    print("Starting preprocessing of files.")
    
    with Pool(num_cores) as p:
        p.map(load_filter_dump, file_list)

    # for file in file_list:
    #     load_filter_dump(file)

    print(f"Finished preprocessing {len(file_list)} files.")

if __name__ == '__main__':

    main()