# coding: UTF-8
import torch
import cv2
import numpy as np
import time
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F

# Depth-Anything-V2モデルの設定
model_name = "LiheYoung/depth-anything-small-hf"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModelForDepthEstimation.from_pretrained(model_name)

lossfunc = F.mse_loss

def load_img_as_tensor(img_path, grayscale=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    img = torch.tensor(img, dtype=torch.float32) / 255.0
    print("Load image from {} shape {}".format(img_path, img.shape))
    return img

def save_tensor(img, path):
    cv2.imwrite(path, (img.cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8))

def output_model_depth(input):
    input = input[..., [2, 1, 0]]
    inputs = processor(images=input.cpu().numpy(), return_tensors="pt", do_rescale=False)

    with torch.no_grad():
        outputs = model(**inputs)
    
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (128, 128))
    depth_min, depth_max = depth.min(), depth.max()
    depth_vis = (depth - depth_min) / (depth_max - depth_min)
    return depth_vis  # 0-1 に正規化されたグレースケール深度マップを返す

def tensor_randomtransform(img, noise_std=0.5, brightness=0.1, contrast=0.1, blur_sigma=0.5):
    noise = torch.randn_like(img) * noise_std
    img = img + noise
    img = torch.clamp(img, 0, 1)
    # brightness
    return img

def adam_adversarial(epoch, ans_image, original_image, alpha, random_transforms):
    start_time = time.time()
    torch_image = original_image.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([torch_image], lr=alpha)
    
    print("ans_image shape", ans_image.shape)
    print("original_image shape", original_image.shape)
    print("torch_image shape", torch_image.shape)

    for i in range(epoch):
        sample = torch.clamp(torch_image, 0, 1)
        output = output_model_depth(sample.detach())
        depth_loss = lossfunc(torch.from_numpy(output), ans_image)
        
        if random_transforms > 0:
            for t in range(random_transforms):
                sample_tr = tensor_randomtransform(sample)
                output_tr = output_model_depth(sample_tr.detach())
                depth_loss += lossfunc(torch.tensor(output_tr), ans_image)
            depth_loss = depth_loss / random_transforms
        
        image_loss = lossfunc(sample, original_image)
        loss = depth_loss + image_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i == 1) or (i % 20 == 0):
            cv2.imwrite("depth{}.png".format(i), (output * 255).astype(np.uint8))
            cv2.imwrite("image{}.png".format(i), (sample.detach().cpu().numpy() * 255).astype(np.uint8))

        elapsed_time = time.time() - start_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), end=" ")
        print("processed {} / {}, loss {:.5f} d_loss {:.5f} i_loss {:.5f}".format(i+1, epoch, loss, depth_loss, image_loss), end="\r")

    print("\n total time : {:.1f} sec".format(time.time() - start_time))
    return torch.clamp(torch_image, 0, 1)

# 画像の初期設定
input_dir = "2.png"
target_dir = "qr_code.png"
save_path = "4.png"
input_image = load_img_as_tensor(input_dir)
target = load_img_as_tensor(target_dir, grayscale=True)

epoch = 600
alpha = 0.01
random_transforms = 0
perturbed_image = adam_adversarial(epoch, target, input_image, alpha, random_transforms)   # w=0.5
save_tensor(perturbed_image, save_path)
print("saved to {}".format(save_path))


