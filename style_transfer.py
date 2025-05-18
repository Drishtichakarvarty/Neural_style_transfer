import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import matplotlib.pyplot as plt

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMAGE LOADER
def load_image(img_path, max_size=512):
    image = Image.open(img_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# DISPLAY/SAVE IMAGE
def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

# GET FEATURES
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# GRAM MATRIX
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# MAIN STYLE TRANSFER FUNCTION
def style_transfer(content_path, style_path, output_path, steps=500, style_weight=1e6, content_weight=1e0):
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)
    target = content.clone().requires_grad_(True).to(device)

    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    for param in vgg.parameters():
        param.requires_grad = False

    style_features = get_features(style, vgg)
    content_features = get_features(content, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    optimizer = optim.Adam([target], lr=0.003)

    for step in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

        total_loss = style_weight * style_loss + content_weight * content_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, Total loss: {total_loss.item():.2f}")

    save_image(target, output_path)
    print(f"Saved stylized image to {output_path}")

if __name__ == '__main__':
    content_file = "content_images/c3.jpg"       # Change this file as needed
    style_file = "style_images/s4.jpg"
    output_file = "output_images/stylized_portrait.jpg"
    style_transfer(content_file, style_file, output_file)
