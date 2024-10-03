import torch
from torchvision import models
from PIL import Image
import numpy as np
import argparse
import json
from torch import nn

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file for category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    
    return parser.parse_args()

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image).float()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    else:
        raise ValueError(f"Model architecture {checkpoint['arch']} is not supported")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], 4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(4096, checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, top_k=5):
    model.eval()
    image = process_image(image_path).unsqueeze_(0)
    
    with torch.no_grad():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
    
    return top_p, top_class

def main():
    args = get_input_args()

    # Load model
    model = load_checkpoint(args.checkpoint)

    # Move model to appropriate device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Predict
    top_p, top_class = predict(args.image_path, model, args.top_k)

    # Map classes to flower names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        labels = [cat_to_name[idx_to_class[idx.item()]] for idx in top_class[0]]
        print(f"Predicted classes: {labels}")
    else:
        print(f"Predicted classes: {top_class}")

    print(f"Probabilities: {top_p}")

if __name__ == '__main__':
    main()
