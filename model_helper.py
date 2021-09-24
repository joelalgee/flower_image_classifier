# python 3.8.5

# Imports here
import os
import torch                                         # 1.7.0
from torch import nn
from torch import optim
from torchvision import models

# Set PyTorch random seed to aid bugfixing
torch.manual_seed(42);

def build_model(arch, hidden_units):
    ''' Configures a pretrained model for predicting flower classes
    '''

    print(f'Building and configuring {arch} model...\n')

    if arch == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.vgg13_bn(pretrained=True)

    # Prevent the pretrained model parameters from further learning
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with an untrained version, tailored for classifying our images into 102 categories
    if arch == 'squeezenet':
        model.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Conv2d(512, 102, kernel_size=(1, 1), stride=(1, 1)),
                                         nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                         nn.LogSoftmax(dim=1))
        model.num_classes = 102
    else:
        model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(in_features=hidden_units, out_features=102),
                                         nn.LogSoftmax(dim=1))
    print(f'{arch} model built and configured.\n')
    if arch != 'squeezenet':
        print(f'{hidden_units} hidden units.\n')

    return model

def load_checkpoint(filepath, gpu):
    ''' Loads a checkpoint and rebuilds the model, device, criterion and optimizer
    '''
    print(f'Loading checkpoint from {filepath}...\n')

    # Load checkpoint data
    checkpoint = torch.load(filepath)
    
    # Create model and restore state
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Work on GPU if requested and available
    if gpu == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Send model to device
    model = model.to(device);

    print(f'Checkpoint at {filepath} loaded.\n')
    
    return model, device

def predict(image, model, device, topk):
    ''' Predicts the class (or classes) of an image using a trained deep learning model.
    '''

    print('Predicting classes...\n')
    
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)
        
        output = model.forward(image)
    
        percentages = torch.exp(output)
        probs, indeces = percentages.topk(topk, dim=1)
        
        probs = probs[0].tolist()
        indeces = indeces[0].tolist()

        # Convert indeces to classes
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indeces]

    model.train()
    
    print('Classes predicted:\n')

    return probs, classes
   
def save_model(model, arch, hidden_units, save_dir, class_to_idx):
    ''' Saves a model checkpoint
    '''

    print(f'Saving model checkpoint at {save_dir}/checkpoint.pth...\n')

    # Create the checkpoint
    checkpoint = {'arch':arch,
                  'hidden_units':hidden_units,
                  'model_state_dict':model.state_dict(),
                  'class_to_idx':class_to_idx
                  }
    if save_dir == '':
        save_dir = arch

    # Create the directory if needed
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    #Save the checkpoint
    torch.save(checkpoint, save_dir + '/checkpoint.pth')

    print(f'Model checkpoint saved at {save_dir}/checkpoint.pth\n')

def train_model(model, learning_rate, epochs, gpu, train_dataloader, valid_dataloader):
    ''' Trains a pretrained model for predicting flower classes
    '''

    print(f'Training model for {epochs} epochs at learning rate {learning_rate}...\n')

    # Work on GPU if requested and available
    if gpu == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Send model to device
    model = model.to(device);

    # Set loss function
    criterion = nn.NLLLoss()

    # Set optimizer and train only the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Set validation interval
    steps_per_valid =25

    # Set up loss tracking
    step = 0
    running_loss = 0

    for epoch in range(1, epochs+1):
        step = 0
        for images, labels in train_dataloader:
            step += 1
            
            # Send batch data tp GPU if available
            images, labels = images.to(device), labels.to(device)
            
            # Set gradients to zero
            optimizer.zero_grad()
            
            # Run batch forwards through model
            output = model.forward(images)
            
            # Calculate loss and gradients
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            #loss_orig = loss.item()
        
            # Optimise weights
            optimizer.step()
            
            # Check model against validation data every 5 batches
            if step % steps_per_valid == 0:
                # Set up validation loss and accuracy tracking
                valid_loss = 0
                valid_acc = 0
                
                # Disable dropout and gradient tracking
                model.eval()
                with torch.no_grad():
                    n_validations = 0
                    for images, labels in valid_dataloader:
            
                        # Send batch data tp GPU if available
                        images, labels = images.to(device), labels.to(device)
                        
                        # Run validation batch forwards through model
                        output = model.forward(images)
                        
                        # Calculate loss
                        loss = criterion(output, labels)
                        valid_loss += loss.item() * len(images)
                        
                        # Add accuracy of validation predictions
                        percentages = torch.exp(output)
                        top_percentage, top_class = percentages.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                        
                        n_validations += len(images)

                    # Calculate loss and accuracy metrics
                    running_loss = running_loss / steps_per_valid
                    valid_loss = valid_loss / n_validations
                    valid_acc = valid_acc / n_validations
                    
                    print(f'Epoch {epoch} Step {step}\n' +
                          f'Running loss {running_loss}\n' +
                          f'Validation loss {valid_loss}\n' +
                          f'Validation accuracy {valid_acc}\n')
                    
                # Enable dropout
                model.train()
                
                # Reset loss tracking
                running_loss = 0

    print(f'Model trained.\nLearning rate {learning_rate}.\n{epochs} epochs.\n')

    return model