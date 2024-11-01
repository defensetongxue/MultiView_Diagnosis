# for each image generate a mask for edge region by analysing the pixel value
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms


def generate_mask(image_path, left_cut=0, right_cut=0, save_path=None):
    if left_cut=='auto':
        pass
        raise ValueError('unImplement parameters')
    img = Image.open(image_path)
    img = np.array(img)/255
    img = np.sum(img, axis=-1)
    mask = np.where(img < 0.26, 0, 1)
    # Get the width of the image
    img_width = img.shape[1]
    # Set left and right areas to zero
    mask[:, :left_cut] = 0  # Set left area to zero
    mask[:, img_width-right_cut:] = 0  # Set right area to zero

    # Convert the mask to an image and save it
    if save_path is not None:
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(save_path)
    return mask


class GammaTransform:
    def __init__(self, gamma_red=0.7, gamma_green=0.9):
        self.gamma_red = gamma_red
        self.gamma_green = gamma_green

    def __call__(self, image):
        red_channel = image[0, :, :] ** self.gamma_red
        green_channel = image[1, :, :] ** self.gamma_green
        transformed_image = torch.stack(
            [red_channel, green_channel, image[2, :, :]])
        return transformed_image


class BilateralFilter:
    def __init__(self, d=10, sigma_color=40, sigma_space=30):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, image):
        image_np = (image.permute(1, 2, 0).cpu().numpy()
                    * 255).astype(np.uint8)
        filtered_image = cv2.bilateralFilter(
            image_np, self.d, self.sigma_color, self.sigma_space)
        filtered_tensor = torch.from_numpy(
            filtered_image).permute(2, 0, 1).float() / 255.0
        return filtered_tensor


class ClaheEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        red_channel = image[0, :, :]
        green_channel = image[1, :, :]
        red_eq = clahe.apply(
            (red_channel.cpu().numpy() * 255).astype(np.uint8))
        green_eq = clahe.apply(
            (green_channel.cpu().numpy() * 255).astype(np.uint8))
        red_tensor = torch.from_numpy(red_eq).float() / 255.0
        green_tensor = torch.from_numpy(green_eq).float() / 255.0
        enhanced_image = torch.stack(
            [red_tensor, green_tensor, image[2, :, :]])
        return enhanced_image


class Enhancer:
    def __init__(self):
        self.enhance_transform = transforms.Compose([
            GammaTransform(),
            BilateralFilter(),
            ClaheEnhancer()
        ])
    def enhanced_image(self,image_path,save_path):
        
        image = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image)
        enhanced_tensor = self.enhance_transform(image_tensor)
        # Save the enhanced image
        enhanced_image_np = (enhanced_tensor.permute(
            1, 2, 0).numpy() * 255).astype(np.uint8)
        enhanced_image = Image.fromarray(enhanced_image_np)
        enhanced_image.save(save_path)
        return save_path

if __name__ == '__main__':
    pass