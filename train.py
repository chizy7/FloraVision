import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    
    return parser.parse_args()

def get_input_size(model):
    # extract th input size from the classifier layer of the pre-trained mdel
    if isinstance(model, models.VGG):
        return model.classifier[0].in_features # VGG like models
    elif isinstance(model, models.ResNet):
        return model.fc.in_features # ResNet like models
    else:
        raise ValueError('Unsupported architecture for input size extraction')

def train_model(args):
    # Load data
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), transform=data_transforms['valid'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    }

    # Load a pre-trained model
    if args.arch == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    elif args.arch == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    else:
        raise ValueError('Unsupported architecture')
        
    # Input size dynamically
    input_size = get_input_size(model)

    # Freeze params
    for param in model.parameters():
        param.requires_grad = False

    # Define the new classifiier
    classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(args.hidden_units, 102),
                               nn.LogSoftmax(dim=1))

    if args.arch == 'vgg16':
        model.classifier = classifier
    elif args.arch == 'resnet18':
        model.fc = classifier

    # Set criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train model
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = args.epochs
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation step
        model.eval()
        accuracy = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                val_loss += criterion(logps, labels).item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Training Loss: {running_loss/len(dataloaders['train'])}, "
              f"Validation Loss: {val_loss/len(dataloaders['valid'])}, "
              f"Validation Accuracy: {accuracy/len(dataloaders['valid']) * 100:.2f}%")
        model.train()  # Set back to training mode after validation

    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'arch': args.arch,
                  'input_size': input_size,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, args.save_dir)
    print(f"Model saved at {args.save_dir}")

if __name__ == '__main__':
    args = get_input_args()
    train_model(args)
