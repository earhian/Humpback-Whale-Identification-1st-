import cv2
import numpy as np
import random
import torch
def random_cropping(image, target_shape=(128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_w, target_h = target_shape
    width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    else:
        start_x = (target_w - width)//2
        start_y = (target_h - height)//2
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    return zeros
def TTA_cropps(image, target_shape=(128, 128, 3)):
    width, height, d = image.shape
    target_w, target_h, d = target_shape
    start_x = (target_w - width) // 2
    start_y = (target_h - height) // 2
    starts = [[start_x, start_y], [0, 0], [ 2 * start_x, 0],
              [0, 2 * start_y], [2 * start_x, 2 * start_y]]
    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        zeros = np.zeros(target_shape)
        zeros[x:x + width, y: y+height, :] = image_
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

        zeros = np.fliplr(zeros)
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

    return images



def random_erase(image,mask, p=0.5):
    if random.random() < p:
        width, height, d = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(5,10)
        b_h = random.randint(5,10)
        image[x:x+b_w, y:y+b_h] = 0
        mask[x:x+b_w, y:y+b_h] = 0
    return image, mask
def random_cropping3d(image, mask, target_shape=(8, 128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_l,target_w, target_h = target_shape
    length, width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
    else:
        start_x = (target_w - width) // 2
        start_y = (target_h - height) // 2
    zeros[:target_l,start_x:start_x+width, start_y:start_y+height] = image
    return zeros
def random_shift(image,mask, p=0.5):
    if random.random() < p:
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        zero_mask = np.zeros_like(mask)
        w = random.randint(0, 20) - 10
        h = random.randint(0, 30) - 15
        zero_image[max(0, w): min(w+width, width), max(h, 0): min(h+height, height)] = \
            image[max(0, -w): min(-w+width, width), max(-h, 0): min(-h+height, height)]
        zero_mask[max(0, w): min(w + width, width), max(h, 0): min(h + height, height)] = \
            mask[max(0, -w): min(-w + width, width), max(-h, 0): min(-h + height, height)]
        image = zero_image.copy()
        mask = zero_mask.copy()
    return image, mask
def random_scale(image, mask, p=0.5):
    if random.random() < p:
        scale = random.random() * 0.1 + 0.9
        assert 0.9 <= scale <= 1
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        zero_mask = np.zeros_like(mask)
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_height, new_width))
        mask = cv2.resize(mask, (new_height, new_width))
        start_w = random.randint(0, width - new_width)
        start_h = random.randint(0, height - new_height)
        zero_image[start_w: start_w + new_width,
        start_h:start_h+new_height] = image
        image = zero_image.copy()
        zero_mask[start_w: start_w + new_width,
        start_h:start_h + new_height] = mask
        mask = zero_mask.copy()
    return image, mask
def change_scale(image, scale=1):
    if 1:
        assert 0.9 <= scale <= 1
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_height, new_width))
        start_w = (width - new_width)//2
        start_h = (height - new_height)//2
        zero_image[start_w: start_w + new_width,
        start_h:start_h+new_height] = image
        image = zero_image.copy()
    return image



def random_flip(image, p=0.5):
    if random.random() < p:
        if len(image.shape) == 2:
            image = np.flip(image, 1)
        elif len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.flip(image, 1)
            image = np.transpose(image, (2, 0, 1))
    return image


def do_gaussian_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = np.random.normal(0,sigma,(H,W))
    noisy = gray + noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + (1-gray) * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def random_angle_rotate(image, mask, angles=[-30, 30]):
    angle = random.randint(0, angles[1]-angles[0]) + angles[0]
    image = rotate(image, angle)
    mask = rotate(mask, angle)
    return image, mask
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

## illumination ====================================================================================

def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha*255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha*image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray  = image * np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
    gray  = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha*image  + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):

    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
		for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table) # apply gamma correction using the lookup table


def do_clahe(image, clip=2, grid=16):
    grid=int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid)).apply(gray)
    lab   = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image

def do_flip_transpose(image, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)



    if type==2: #rotate180
        image = cv2.flip(image,-1)


    if type==3: #rotate270
        image = image.transpose(1,0,2)
        image = cv2.flip(image,0)



    if type==4: #flip left-right
        image = cv2.flip(image,1)


    if type==5: #flip up-down
        image = cv2.flip(image,0)

    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)



    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)



    return image

def bgr_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
def do_flip_transpose_4(image, type=0):
    #choose one of the 8 cases

    if type==0: #rotate180
        image = cv2.flip(image,-1)


    if type==1: #flip left-right
        image = cv2.flip(image,1)


    if type==2: #flip up-down
        image = cv2.flip(image,0)


    return image
