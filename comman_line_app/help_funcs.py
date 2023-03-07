# Importing the required packages 
import torch
from torch import nn, optim
from torch.nn import Sequential
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# loading the data into data loaders 
# and performing the required normalizations
# for training, valid, and testing data

def load_data(data_dir):
    """
    a function that takes the data directory as input,
    then transforms the data to fit the pretrained models,
    and load the data with ImageFolder.
    Then it returns the data into data loaders to help in training.
    
    args:
         data_dir (str)
        
    returns:
         trainloader, validloader, testloader
    
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining a normalizer for the image color channels
    # to fit the pre-trained networks
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Defining transformers for the training set.
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalizer])
    
    # Defining transformers for the validation and test sets.
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalizer])
        
    # Loading the training set with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms) 
    
    # Loading the validation set with ImageFolder
    valid_set = datasets.ImageFolder(valid_dir, transform=test_transforms) 
    
    # Loading the test set with ImageFolder
    test_set = datasets.ImageFolder(test_dir, transform=test_transforms) 
        
    # Defining DataLoaders

    # training dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    
    # validation dataloader
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64)
    
    # training dataloader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    
    class_to_idx = train_set.class_to_idx
    
    return train_loader, valid_loader, class_to_idx



# downloading the pretrained model
def pretrained_model(arch='resnet152', hidden_units=512):
    """
    a function that downloads a pretrained model from
    torchvision.models, modifies its architecture,
    and use it as an image classifier after training.
    
    args:
        arch: (str) the pretrained model architecture 
               uses resnet152 as defult
        hidden_units: (int) number of units in the hidden
                       layer of the model classifier
    returns: 
        model: the model arch after modefication to fit 
        the possible classes
    """
    
    arch = arch.lower()
    

    # modifying the model classifier arch
    # to fit the possible classes
    
    if arch == 'resnet152':
        # downloading the model
        model = models.resnet152(pretrained=True)
        
        # freezing the features parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # defining classifier parameters to fit
        # the 102 categories for our classifier output
        model.fc = nn.Sequential(nn.Linear(2048, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1))

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(4096, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
    return model


# A training function to train the model
# building a back propagation algorithm to..
# train the classifier parameters and tracking..
# the loss and accuracy on the validation set

# Because the following code will take a lot of time
# we need to keep the session active so we use 
# active_session from workspace_utils module

def train_model(data_dir, model, arch, device,  lr=0.001, epochs=1):
    """
    a function that trained the choosen pretrained model
    on the data in the provided directory. The training uses
    NLLLoss - as criterion - and Adam optimizer.
    The code runs in CPU or GPU according to the device arg.
    
    args:
        data_dir (str): the directory of the data
        model (str): the pretrained model used in training
        device (str): the device in which the code runs [cpu or gpu]
        arch (str): the model architecture
        lr (float): the learning rate used in the model optimizer
        epochs (int): the number of epochs used in model training  
        
    returns:
        None: it trains the model only
    """
    
    # loading the data in data loaders 
    train_loader, valid_loader, class_to_idx = load_data(data_dir)
    

    # moving the model to the device
    model = model.to(device)
    
    # defining our criterion as NLLLoss 
    criterion = nn.NLLLoss()
    
    # using Adam optemizer to train the..
    # model calssifier parameters
    if arch == 'vgg16':
        classifier = model.classifier
    elif arch == 'resnet152':
        classifier = model.fc
    
    optimizer = optim.Adam(classifier.parameters(), lr)
    
    # we save mapping of classes to indices from an image set
    model.class_to_idx = class_to_idx
    
    # setting up the active session mode
    # to avoid workspace shutdown during model training
    # as the training can take a lot of time
    from workspace_utils import active_session

    with active_session():

        for e in range(epochs):
            
            step = 0
            train_loss = 0

            # iterating across train_loader
            for images, labels in train_loader:
                step += 1
                # moving images and labels to the device mode
                images, labels = images.to(device), labels.to(device)
            
                optimizer.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
                #if step % 10 == 0:
                    #print(f"epoch: {e}  {step} steps..." )
            # -------------------------
            # iterating across the valid set to calculate Loss and accuracy
            
            valid_loss = 0
            accuracy = 0
            
            # turning off the auto grad feature to fasten the code
            # and moving to evaluating mode
            with torch.no_grad():
                model.eval()
                for images, labels in valid_loader:
                    # moving images and labels to the device mode
                    images, labels = images.to(device), labels.to(device)
                    
                    log_probs = model(images)
                    valid_loss += criterion(log_probs, labels)
                    probs = torch.exp(log_probs)
                    
                    # finding the top class in each of the probs
                    top_prob, top_class = probs.topk(1, dim=1)

                    # matches between top_class and labels
                    matches = top_class == labels.view(top_class.shape)
                    
                    accuracy += torch.mean(matches.type(torch.FloatTensor))

                print(f"in epoch {e+1} after {step} steps train_loss = {train_loss/len(train_loader):.3f},  "
                      f"valid_loss = {valid_loss/len(valid_loader):.3f}  "
                      f"valid accuracy = {accuracy/len(valid_loader):.3f}")
                    
            # moving to training mode
            model.train()
            train_loss = 0
            accuracy_train = 0

            
# saving the checkpoint
def save_checkpoint(save_dir, model, arch, hidden_units):
    """
    a function to save the traied model to a checkpoint
    args:
        save_dir (str): the directory at which you need to save
    
    returns: it only save the model.
    """
    # moving to cpu
    model.cpu()
    
    # saving the model arch and the class_to_idx
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'arch': arch,
                'hidden_units': hidden_units}, 
               f = save_dir)
    

# A function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    """
    A function that loads the saved checkpoint 
    and rebuilds the trained model with the same
    architecture and the same classes to indices
    
    args:
        checkpoin_path (str): the path to the saved checkpoint
        
    returns:
        model: the trained model.
    """
    
    # firstly we load the saved checkpoint dict
    checkpoint_dict = torch.load(checkpoint_path)
    
    # model architecture
    arch = checkpoint_dict['arch']
    # number of units in the classifier hidden layer
    hidden_units = checkpoint_dict['hidden_units']  
    
    # Download the model and modify the classifier architecture
    
    if arch == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Sequential(nn.Linear(2048, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(4096, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, 102),
                                         nn.LogSoftmax(dim=1))
        
    # loading the saved state_dict and class_to_idx to the model
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
  
    return model


# Image processing function
def process_image(img_path):
    """ 
    A function that scales, crops, and normalizes a PIL image
      for a PyTorch model.
      
      args:
          img_dir (str): The image path
       
      returns:
          img: A numpy array representing the processed image.
    """
    
    # opening the image to process
    img = Image.open(img_path)
    
    # resizing the image to 256 pixels for the shortest side
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # crop out the center 224x224 portion of the image
    w, h = img.size  # getting the image size
     
    # defining the dimensions surrounding the croped image 
    left = (w - 224) / 2
    upper = (h - 224) / 2
    right = (w + 224) / 2
    lower = (h + 224) / 2
    img = img.crop((left, upper, right, lower))
    
    # normalizing the image using np.array
    np_image = np.array(img) / 255
    
    # mean normalization
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ( np_image - means ) / std
    
    # transposing the image array so that
    # the color channel can be the first dimension
    img = np_image.transpose((2,0,1))
    
    return img


# Prediction function
def predict(img_path, checkpoint, top_k, cat_to_name, device):
    """
    Predict the class (or classes) of an image 
    using a trained deep learning model checkpoint.
    
    args:
        img_path (str): The image path.
        checkpoint (str): The path to a saved model checkpoint.
        top_k (int): The number of top_k classes
        cat_to_name (dict): A dictionary mapping categories to names
        device: The device used to run the code
        
    returns:
        top_names (list): The name (or names) of the top_k classes
        top_probs (list): The probabilities of the top_k classes
    """
    
    # loading the model
    model = load_model(checkpoint)
    model = model.to(device)
    
    # image processing
    img = process_image(img_path)
    
    # converting the image numpy array to a torch tensor
    # and moving it to the device
    img = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    # applying the model that gives...
    # log probabilities then getting the probs
    log_probs = model(img.unsqueeze_(0))
    probs = torch.exp(log_probs)
    
    # finding the topk probs and indices 
    top_probs, top_indices = probs.topk(top_k)
    
    top_probs, top_indices = top_probs[0].tolist(), top_indices[0].tolist()
    
    # rounding the top_probs
    top_probs = [round(prob, 3) for prob in top_probs]
    # converting the topk indices into classes
    # firstly, we invert the class_to_idx dict
    idx_to_class = {i:c for c, i in model.class_to_idx.items()} 
    
    # the topk classes
    top_classes = [idx_to_class[i] for i in top_indices]
    
    # converting the classes into names
    top_names = [cat_to_name[c] for c in top_classes]
    
    return top_names, top_probs

