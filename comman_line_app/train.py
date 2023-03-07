#importing required packages
import argparse
import os
import torch
import help_funcs   # a python file having the helping functions

# getting command line inputs
parser = argparse.ArgumentParser()

# adding positional argument for data directory
parser.add_argument('data_dir', 
                    help='the root directory having training and testing data')

# adding optional arguments..
# for the model architecture
parser.add_argument('--arch', 
                    help='the pretrained model architecture ',
                    choices= ['vgg16', 'resnet152'],
                    default='vgg16')

# directory to save checkpoints
parser.add_argument('--save_dir',
                    help='the directory at which you need to save the checkpoint')

# setting hyperparameters
parser.add_argument('--learning_rate',
                    help='(float) set the learning rate for GD algorithm (0.01, 0.001, 0.0001 ...)',
                    type=float,
                    default=.001)

parser.add_argument('--hidden_units',
                    help='(int) number of units in the hidden layer of model classifier',
                    type=int,
                    default=512)

parser.add_argument('--epochs',
                    help='(int) number of training epochs',
                    type=int,
                    default=1)

parser.add_argument('--gpu',
                    help='move the training to GPU device to fasten the process',
                    action='store_true')

# parsing the arguments.
args = parser.parse_args()

data_dir = args.data_dir.strip()    # training data directory

# the model architecture
arch = args.arch.strip()
while arch not in ['vgg16', 'resnet152']:
    """ check that the entered arch is one of the available"""

    print('Sorry, the entered model architecture should be (vgg16 or resnet152)')
    arch = input('Please enter the model architecture again:' ).strip().lower()


# checkpoint directory
if args.save_dir:
    save_dir = os.path.join(args.save_dir, 'my_checkpoint.pth')
else:
    save_dir = 'my_checkpoint.pth'
 
# learning rate
lr = args.learning_rate

# number of units in model classifier hidden layer
hidden_units = args.hidden_units
 
# number of training epochs
epochs = args.epochs

# training in GPU if it is avialable and the user choosed it
device = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('There is no GPU device available, so, CPU will be used..')



# building the model
model = help_funcs.pretrained_model(arch, hidden_units)

# training the model on the data in data_dir
help_funcs.train_model(data_dir, model, arch, device,  lr, epochs)

print('Ok. Do you want more training?')

check = input('Please inter only [yes or no] :').lower().strip()

while check == 'yes':
    epochs = 0
    while epochs == 0:
        try:
            epochs += int(input('How many epochs? (Please inter an int.): '))
        except:
            print('Invalid input. Try again!')
    
    # training the model
    help_funcs.train_model(data_dir, model, arch, device,  lr, epochs)
    
    print('Ok. Do you want more training?')

    check = input('Please inter only [yes or no]:').lower().strip()

print('Ok. Saving the checkpoint at:  {}'.format(save_dir))
print('......')

# saving the checkpoint
help_funcs.save_checkpoint(save_dir, model, arch, hidden_units)