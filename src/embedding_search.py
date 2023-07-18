import os
import json
import numpy as np
import pandas as pd 

from paths import *
from tqdm import tqdm
from annoy import AnnoyIndex
from preprocessing import lemmatize
from multiprocessing import Pool, cpu_count
from os.path import join as pjoin, isdir
from json_load_write import load_json_data
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

## unfinished
def embed_text(text, model, file_name: None, save_path: None):
    pool = model.start_multi_process_pool()

    embeddings = model.encode_multi_process(sentences=text, 
                                            pool=pool)

    # todo: normalize embeddings to improve performance

    if (file_name and save_path):
        output_file_path = pjoin(save_path, file_name.split(".")[0] + ".npy") 
        np.save(output_file_path, embeddings)

    else: 
        return embeddings

# unfinished
def create_index(embeddings, file_name = None, save_path= None): 
    narratives_search_index = AnnoyIndex(np.array(embeddings).shape[1], 'manhattan')

    for index_embedding, embed_value in enumerate(embeddings):
        narratives_search_index.add_item(index_embedding, embed_value)

    narratives_search_index.build(n_trees = 20, n_jobs = -1)

    if (file_name and save_path):
        output_file_path = pjoin(save_path, file_name.split(".")[0] + ".npy") 
    
        narratives_search_index.save(f'2018-01_narrative_search_index.ann')

    return narratives_search_index

def call_lemmatize(data_series, dataframe):
    num_cores = 8

    pool = Pool(num_cores)
    results = pool.imap(func = lemmatize, 
                        iterable = data_series, 
                        chunksize = data_series.size // num_cores)
    
    pool.close()
    pool.join()

    print("Lemmatization complete.")

    dataframe["lemmatized_selftext"] = np.array(list(results))


def main():

    # model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

    # query_embedding_path = pjoin(DECOMPRESSED_SUMBISSIONS, 'csv', 'embeddings',  'old-narrative-queries_embeddings.npy')
    # query_index_path = pjoin(DECOMPRESSED_SUMBISSIONS, 'csv', 'embeddings', 'old-narrative-queries_index.ann')
    
    # if (not os.path.exists(query_embedding_path) and not os.path.exists(query_index_path)):
    #     oldnarrative_queries = list(json.load(open("new_local_data.json", encoding="utf-8"))["File"]) 
    #     tokenized_queries = list(map(lemmatize, oldnarrative_queries))

    #     embed_text(tokenized_queries, model, "old-narrative-queries.json", pjoin(DECOMPRESSED_SUMBISSIONS, 'csv', 'embeddings'))

    # else: 

    #     query_embeddings = np.load(query_embedding_path)
    #     query_search_index = AnnoyIndex(query_embeddings.shape[1], 'manhattan')
    

    csv_directory = os.listdir(pjoin(DECOMPRESSED_SUMBISSIONS, "csv"))


    for file_name in os.listdir(csv_directory):
        if (not isdir(pjoin(DECOMPRESSED_SUMBISSIONS, "csv", file_name))):
            print(f"Processing file {file_name}.")

            df = pd.read_csv(pjoin(DECOMPRESSED_SUMBISSIONS, "csv", file_name))
            call_lemmatize(df["selftext"], df)
            df.to_csv(pjoin(DECOMPRESSED_SUMBISSIONS, "csv", file_name.split(".") + "_lemmatized" + ".csv"), index=False)
            
            print(f"File {file_name} processed.")

if __name__ == "__main__":
    main()
