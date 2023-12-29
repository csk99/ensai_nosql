# IMAGE SEARCH APP



## Project Intro
<p> The advent of the internet revolutionized the way we access information through potent search engines such as Google, Bing, and Yandex. With just a few keywords, we can swiftly locate web pages pertinent to our queries. As technology, particularly AI, advances, many search engines now facilitate online image searches.

Various techniques for image searching have emerged, including:

* image search by metadata:
Here, the search is  not based on the image itself but rather on the metadata following the image like (keywords, text, filename,date etc.) <br> 
* image search based on image content:
This approach uses, state of the art computer vision techniques to extract shape, colour, any relevant features from an image. This is the technique we are going to study.


In this project, we will use a pre-trained Convolutional Neural Network (CNN) to extract valuable features from the images. This methodology, a key component of content-based image search, provides the following benefits:<br> 

* CNN are robust:
CNN have proven to be very powerful to extract key features from an image.
* CNN can reduce dimension:
The CNN output typically represents a condensed, relevant representation of the image often called **feature map or embedding or vectors**, as not every pixel holds significant information. This condensed representation often has smaller dimensions.


In summary, in this study we will like to answer the following question:
**Are two similar images associated embedding are still similar?**
</p> 
### Technologies / Frameworks used 
* ![Static Badge](https://img.shields.io/badge/Python-3.8-green)
* ![Static Badge](https://img.shields.io/badge/Pinecone-2.2-green)
* ![Static Badge](https://img.shields.io/badge/keras-2.13-green)

## Project Description
We have use a subset (10 actions) of [UCF101](https://www.crcv.ucf.edu/data/UCF101.php). Our computer vision model has been trained to be  able to recognize the following action:
* ApplyEyeMakeup       
* BabyCrawling         
* Biking               
* Billiards            
* BrushingTeeth        
* PizzaTossing         
* PushUps              
* Swing                
* TrampolineJumping    
* Typing


## Working principle
The following pictures explains how our models takes in raw video data and output its associated activity or action label.
<img src="principle.png" alt="principle">

## Running time
In this project, we are handling videos, which pose some challenges in terms of compution especially when reading videos, capturing frames and extracting features via a CNN.
We tried to leverage the power of parrallel computing when running our code so that everything runs as fast as possible on multiple  CPU cores via multithreading.<br> 
*Note: If possible, run this project on a GPU powered environment for faster computations.*





## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. You can download raw data from here Raw Data is being kept [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar) .

    *Choose the activities videos folders you're interested in (or everything) and put them inside the dataset folder*
    *Note: Be aware that the whole dataset of UCF is about 6.5Go*
    
3. Create a virtual env in the project folder (for help see this [tutorial](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/))
4. Run the following command to install the necessary packages
* For linux users:
```
pip3 install -r requirements.txt
```
* For windows users:
```
pip install -r requirements.txt
```


5. Open the Har.ipynb notebook and run the cells.<br> 
*Please choose the python virtual environment you created previously*



