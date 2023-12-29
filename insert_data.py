import glob
import tensorflow as tf
import argparse
import time

import pinecone
from utility import preprocessing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(
    prog="Insert image embeddings into pinecone index",
    epilog="Insertion done")


key = "c5fe733e-d73d-429c-b2a2-b7d516518ec7"

parser.add_argument("-key",dest="key",
                    help="api key associated to your pinecone account",
                    required=True)
parser.add_argument("-env",dest="env",
                    help="your pinecone environment",
                    required=True)
parser.add_argument("-metric",
                    dest="metric",help="similarity metric",
                    choices=['cosine',"euclidean"],required=True,default="cosine")

parser.add_argument("-index",
                    dest="index",help="name of index to be created in pinecone",
                    default="images")
args= parser.parse_args()

KEY = args.key
ENV = args.env
METRIC = args.metric
INDEX =args.index
FILES_NAMES = glob.glob("data/data_batch_*") #list of all the files containing the images




#connection to your pinecone env
pinecone.init(api_key=KEY, environment=ENV)

#create an index if not exist already
try:
        pinecone.create_index(INDEX, dimension=512, metric=METRIC)
except Exception:
        print(f"Index: {INDEX} already exists")
t1 = time.time()
print('\n')
for file in tqdm(FILES_NAMES,desc="reading each file"):
        print("\n")
        preprocessing.process_file(file,INDEX)
print('\n')
print('\n')
print(f'Insertion is done and  took {(time.time() - t1) / 60} min')