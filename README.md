# Image-Classifier-Model
This is my graduation project at Udacity AI Programming with Python Nanodegree. 
Along the learning journey, there have been many challenges and valuable learning experiences.
### I have learned a lot about:
- Python programming (with a complete lesson about intro to OPP ) for AI;
- Numpy, Pandas, Matplotlib, and Seaborn;
- Linear algebra, and calculus essentials for Neural Networks;
- Neural Networks, Gradient Descent, and model training;
- Deep learning with PyTorch.

After finishing the nanodegree content, I have implemented the acquired knowledge in a real project that has been reviewed by Udacity helpful reviewers.

The project has been about creating an images' classifier program to classify flower images according to the category of the flower. I managed to complete it with about 90% accuracy.

I -also- has built a Python image classifier application that can be run on the command line. It trains a deep learning model in an images' dataset and uses the trained model to predict the class of a new image.

### The project contains:
###### 1) The ipynb file `Image_Classifier_nb` is the the image classifier notebook,
###### 2) The folder `comman_line_app` contains the code files for the Python image classifier application:

**- `train.py` is the code of model training on a data set of images classified into (train and validation datasets)**
- It takes a positional argument of `data_directory`,
- Optional arguments of `--save_dir` (directory for the checkpoint after training), `--arch` (the training model architecture), `--learning_rate`, and `--gpu` (to implement the code in a GPU device if avialable)

**- `predict.py` is the code of predicting image classification using the trained model checkpoint**
- It takes positional arguments of `image path` and `checkpoint path`,
- Optional arguments of `--top_k` (top k most likely classes), `--category_names` (a path to a jason file containing mapping of categories to real names), `--gpu` (to implement the code in a GPU device if avialable)

###### 3) `flowers` dataset

###### 4) A code to make the session active
