
Challange details:
https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/

Relevant classes:
    download - to download the images and save them to disc
    mf_dataset - to load the dataset
    explore - look at images and their labels based on ID
    mf_trainer - does all the work (image to feature vectors, training,
    validation, evaluation)
    
TODO:
- add FTN network
- use ImageAI object detection to preprocess images by selecting only persons where relevant
- look into adaptive treshold selection for the network predictions.

Current method:

Multilabel classification.

Pretrained Neural Net for feature extraction.

The net parameters should be trained (transfer learning) to extract
features from the current dataset.

The classifier takes the feature vector (1000D vector) as input and tries to fit the 
feature vector with the labels.

Ideas:

Neural net preprocessing extract the clothes, remove the face,
make the image black and white, remove the background

Machine learning classification:
Multiply the images that have few labels to represent only those labels.
Balance the label representation that is fed to the classifier.

Remove the neural net preprocessing for the 1000D feature extraction and 
try the sklearn preprocessing.
