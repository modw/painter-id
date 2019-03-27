# Technical Art History - Building a Classifier to Differentiate Painters
This repository contains key data processing, analysis and modeling files for the Technical Art History project. The data itself can't be shared while the project is ongoing.

## Project and Objective
The objective of this portion of the project is to create a classifier to differentiate artist based on profile measurements of their paintings. The project is an ongoing collaboration between the departments of Art History and Physics at Case Western Reserve University, MORE Center, Cleveland Institute of Art and the Cleveland Museum of Art.

## Data Processing Basics
A single measurement is of the order of 2000x2000 pixels. So to make that into a trainable set we have to split it into small patches (of order 50x50). The same artist might have several different paintings, which adds to the training set nicely and controls for overfitting naturally.

Here are the basics of the data processing pipeline:
* Transform 3d profile data into a 2d numpy array.
* Split the array into patches.
* Offset the patches.
* Prepare X, y arrays for Keras.
* Perform the train-test split.
* Scale data and save scaler for future validation.

## Main Files

* `data_processing.py`: Deals with the data pre-processing. Mainly, `processing_pipeline` takes the painting file addresses as input and outputs X,y patches. `Scaler` is a convenience Class wrapper that fits and transforms. It also contains methods to save and load saved scalers.

* `Visualizing network.ipynb`: This is the environment where the visualization functions are being developed.

* `Keras - CNN Models.ipynb`: This is the environment where Keras models are being defined. Later they will be moved to a .py file so that they can be imported as needed.

* `Training - 2019 data - v1.ipynb`: This is one example of the implementation of processing, modeling, and some data viz.