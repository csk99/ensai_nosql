#Imports

import streamlit as st
import pinecone
import matplotlib.pyplot as plt
import argparse
from streamlit_option_menu import option_menu
from utility import preprocessing
from PIL import Image

parser = argparse.ArgumentParser(
        prog="Image Search App")
parser.add_argument("-key",dest="key",
                    help="api key associated to your pinecone account",
                    required=True)
parser.add_argument("-env",dest="env",
                    help="your pinecone environment",
                    required=True)
parser.add_argument("-index",
                    dest="index",help="name of index to be created in pinecone",
                    default="images")
parser.add_argument('-k',
                    dest='k',help="top k similar embeddings to fetch",
                    default=3,type=int)

args = parser.parse_args()
#define some useful variables to use
KEY = args.key
ENV = args.env
INDEX =args.index
TOP_K = args.k
#
import sys
print(sys.argv[0])
def menu(KEY):
        pinecone.init(api_key=KEY, environment = ENV)
        index = pinecone.Index(INDEX)
        uploaded_file = st.file_uploader(
                "Choose an image to query", accept_multiple_files=False)
        if uploaded_file is not None:
                img = plt.imread(uploaded_file)
                st.write("**Query results :**")
                ##extract features
                img_features = preprocessing.feature_extractor(
                        img.reshape(
                                1,preprocessing.IMG_SIZE,
                                preprocessing.IMG_SIZE,
                                3))[0].numpy().tolist()
                ###pinecone
                try:
                        result = index.query(
                        vector=img_features,
                        top_k=TOP_K,
                        include_metadata=True
                        )
                except :st.experimental_rerun()
                
                result_matches = result['matches']
                fig1,axes = plt.subplots(1,TOP_K+1)
                plt.tight_layout()
                axes[0].imshow(img)
                axes[0].title.set_text("Original image")
                axes[0].set_xlabel(uploaded_file.name.split(".")[0])
                for i in range(0,len(result_matches)):
                        picture,label = preprocessing.id_to_image(
                                result_matches[i-1]['id'])
                        
                        axes[i+1].imshow(picture)
                        axes[i+1].title.set_text(f"score: {result_matches[i-1]['score']:.3f}")
                        axes[i+1].set_xlabel(label)
                st.pyplot(fig1)

        else:st.write("Please select an image")


##app coding part
st.title("Image Search App")
menu(KEY)

       
#init the connection to the pinecone database
