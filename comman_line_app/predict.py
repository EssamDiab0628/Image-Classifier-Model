#importing required packages
import argparse
import torch
import help_funcs   # a file having helping functions
import json

# getting command line inputs
parser = argparse.ArgumentParser()

# adding positional argument for image path
parser.add_argument('img_path', 
                    help='The path of the image we require to predict its class')

# adding positional argument for checkpoint path
parser.add_argument('checkpoint',
                    help='The path of the saved trained model used for prediction')

# adding optional argument for topK (most likely classes)
parser.add_argument('--top_k',
                    help='The number of most likely classes (topk)',
                    type=int,
                    default=1)

# adding an optional arugument for path of the json 
# file having mapping of categories to real names
parser.add_argument('--category_names',
                    help='The path of the json file having mapping of categories to real names',
                    default='cat_to_name.json')

parser.add_argument('--gpu',
                    help='Inference using a GPU device',
                    action='store_true')

# parsing the arguments.
args = parser.parse_args()

img_path = args.img_path.strip()    # The image path

checkpoint_path = args.checkpoint.strip()    # The chechpoint path

top_k = args.top_k   # Number of top_k categories 

cat_names_path = args.category_names   # The path of category names json file

# predicting in GPU if it is avialable and the user choosed it
device = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('There is no GPU device available, so, CPU will be used..')

    

# downloading the category to name json file
with open(cat_names_path, 'r') as f:
    cat_to_name = json.load(f)

    

# Predicting image class_name and its probability prediction
top_names, top_probs = help_funcs.predict(img_path, checkpoint_path, top_k, cat_to_name, device)

print('The predicted class (or classes): ', top_names,
      '\n The probability (or probabilities): ', top_probs)

