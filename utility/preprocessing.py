import pickle5 as pickle
import pinecone
import time
import tensorflow as tf
import keras
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

#define some global varibales to be use in the entire program
IMG_SIZE = 32
COLORS_CH = 3
NUM_THREADS = os.cpu_count() - 1 # the number of threads
CLASSES_DICT = {
        0:"Airplane",1:"Automobile",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",
        7:"Horse",8:"Ship",9:"Truck"
}

#load a pretrained VGG16 CNN to extract features from images
feature_extractor = tf.keras.applications.VGG16(
            weights="imagenet",
            include_top=False, 
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, COLORS_CH),
)

def load_data(file: str) -> Any:
    """
    Load CIFAR-10 data from a pickle file.

    Args:
    - file (str): The path to the pickle file.

    Returns:
    - Any: The loaded data.
    """
    with open(file, mode='rb') as f:
        data = pickle.load(f, encoding="bytes")
    return data


def id_to_image(id: str) -> np.ndarray:
    """
    Convert an ID to an image array.

    Args:
    - id (str): The identifier used to fetch the image.

    Returns:
    - np.ndarray: The image data reshaped and transposed.
    """
    file = "data/data_batch_" + id[6]
    print(os.getcwd())
    print(file)
    data = load_data(file)
    img = data[b'data'][int(id[8:])].reshape(
         COLORS_CH, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
    label = CLASSES_DICT[data[b'labels'][int(id[8:])]]
    return img,label


def process_file(file: str, index:str,feature_extractor: keras.Model=feature_extractor, classes_dict: Dict=CLASSES_DICT) -> None:
    """
    Process a file to convert images to embeddings and insert it in pinecone.

    Args:
    - file (str): The file to be processed.
    - feature_extractor (keras.Model): Extractor used for feature extraction.
    - classes_dict (Dict): Dictionary mapping labels to classes.

    Returns:
    - None
    """
    data = load_data(file)
    images = data[b'data'].reshape(
        data[b'data'].shape[0], COLORS_CH, IMG_SIZE, IMG_SIZE).transpose(0, 2, 3, 1)

    print("Converting pictures to embedding...")
    images_features = feature_extractor(images).numpy().tolist()
    print("done")
    #for each insert, we create a tuple having 3 elements (a,b,c)
    #a:   Image ID ,b: image embedding, c: metadata: original image label (e.g truck, airplane...)
    #vector is a list that contains all the tuples and is of size 10000 * 5 
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        vectors = list(
            executor.map(
            lambda i: ('batch_' + str(data[b'batch_label'])[-7] + '_' + str(i),
            images_features[i],
            {"label": classes_dict[data[b'labels'][i]]})
            ,list(range(data[b'data'].shape[0])))
                     )
        

    """vectors = []   
    for i in tqdm(range(data[b'data'].shape[0])):
        vectors.append(
            (
                'batch_' + str(data[b'batch_label'])[-7] + '_' + str(i),
                images_features[i],
                {"label": classes_dict[data[b'labels'][i]]}
            )
        )"""

    index = pinecone.Index(index)
    index.upsert(vectors=vectors, batch_size=200)




