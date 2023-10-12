import os
import json
import numpy as np
import pandas as pd 
import csv
import logging

from paths import *
from time import time
from tqdm import tqdm
from datetime import datetime
from annoy import AnnoyIndex
from src.collection_pipeline.utils.preprocessing import lemmatize
from multiprocessing import Pool, cpu_count
from os.path import join as pjoin, isdir
from Archive.json_load_write import load_json_data
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')   
pd.set_option('display.max_colwidth', None)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename='extraction_pipeline.log', filemode='a', level=logging.INFO)

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

    narratives_search_index.build(n_trees = 20, n_jobs = 4)

    if (file_name and save_path):
        output_file_path = pjoin(save_path, file_name.split(".")[0] + ".npy") 
    
        narratives_search_index.save(f'2018-01_narrative_search_index.ann')

    return narratives_search_index

def call_lemmatize(data_series, dataframe, file_name = None):
    if file_name: 
        logging.info(f"Lemmatization started for {file_name}.")
    else: 
        logging.info(f"{datetime.now()}: Lemmatized started for unnamed file.")

    num_cores = 60

    pool = Pool(num_cores)
    results = pool.imap(func = lemmatize, 
                        iterable = data_series, 
                        chunksize = data_series.size // num_cores)
    
    pool.close()
    pool.join()

    if file_name: 
        logging.info(f"Lemmatization complete for {file_name}.")
    else: 
        logging.info(f"Lemmatized complete for unnamed file.")

    try:
        dataframe["lemmatized_selftext"] = np.array([result for result in results])

    except Exception as e:
        logging.error(f"Error in lemmatization for {file_name}: {e}")
        raise e

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
    

    csv_directory = os.listdir(pjoin(DECOMPRESSED_SUMBISSIONS))
    csv_directory.sort()

    for file_name in csv_directory:
        file_path = pjoin(DECOMPRESSED_SUMBISSIONS, file_name)
        if (not isdir(file_path)):
            lemmatized_file_path = pjoin(DECOMPRESSED_SUMBISSIONS, file_name.split(".")[0] + "_lemmatized.csv")

            if ((not os.path.isfile(lemmatized_file_path)) and ('lemmatized' not in file_name)): 
                st = time() 
                logging.info(f"Processing file {file_name}.")

                df_chunks= pd.read_csv(file_path, 
                                       usecols= ['id', 'selftext', 'title', 'subreddit', 'permalink', 'url'], 
                                       chunksize=75_000)    
                
                chunk_array = []
                for index,chunk in enumerate(df_chunks):
                    call_lemmatize(data_series=chunk["selftext"], dataframe=chunk, file_name = f"{file_name}, chunk_{index}")
                    chunk_array.append(chunk)

                df = pd.DataFrame()

                for chunk in chunk_array: 
                    df = pd.concat([df, chunk])

                df.to_csv(lemmatized_file_path, index=False)
                
                logging.info(f"File {file_name} processed. Took {(time() - st)//60} minutes for {df.shape}")
            else: 
                logging.info(f"File {file_name} already lemmatized.")

    
if __name__ == "__main__":
    main()