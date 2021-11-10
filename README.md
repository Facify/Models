# Models

## About

Facify consists of Machine Learning and Neural Network models for faces and identifying their age and various other details.

## Installation

-   Clone the repository using `git clone git@github.com:Facify/Models.git `.
-   Install the required libraries and packages.
-   Download the datasets require in dataset folder.
-   Ready to run the code in jupyter notebook.

## Getting the datasets

### UTK_dataset

-   To download utk dataset go click on the following link [UTK_dataset](https://susanqq.github.io/UTKFace/).
-   Add the unzip the dataset and copy it to the dataset folder in the repository.
-   There are currently 3 models for UTK dataset as described below
    
    - CNN_UTK_Model_1 is a shallow CNN model that uses regression to predict age of the facial image. 
     ![CNN_UTK_Model_1](https://github.com/Facify/Models/blob/main/models/cnn_utk_model_1.png)
    - CNN_UTK_Model_1_AgeRange is a shallow CNN model that uses regression to predict age range(range of 5) of facial image. 
     ![CNN_UTK_Model_1](https://github.com/Facify/Models/blob/main/models/cnn_utk_model_1_age-range.png)
    - CNN_UTK_Model_1_AgeRange_Classification is a shallow CNN model that uses classification to predict age range(range of 5) of facial image. <br>
     ![CNN_UTK_Model_1](https://github.com/Facify/Models/blob/main/models/cnn_utk_model_1_age-range-classification.png)
- The Regression models have been observed to provide the best results.
