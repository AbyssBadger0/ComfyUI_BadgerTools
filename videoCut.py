import os
import shutil
import torch
import open_clip
import numpy as np
import cv2
from sentence_transformers import util
from PIL import Image
from skimage import metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)


def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def tensor_to_cv2(image):
    i = 255. * image.cpu().numpy()
    pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    image_array = np.array(pil_image)
    image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array_bgr


def SSIM(img0, img1):
    image0 = tensor_to_cv2(img0)
    image1 = tensor_to_cv2(img1)
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
    return round(ssim_score[0], 2) * 100


def generateScore(img0, img1):
    image0 = tensor_to_cv2(img0)
    image1 = tensor_to_cv2(img1)
    image0 = imageEncoder(image0)
    image1 = imageEncoder(image1)
    cos_scores = util.pytorch_cos_sim(image0, image1)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score


def getCutList(images, min_frame, max_frame):
    cutList = []
    resList = []
    indexList = []
    i = min_frame - 1
    num = 0
    while i < len(images) - 1:
        num += 1
        img0 = images[i]
        img1 = images[i + 1]
        res = generateScore(img0, img1)
        print("切割画面（" + str(i + 1) + "/" + str(len(images)-1) + ")" + str(res))
        resList.append(res)
        indexList.append(i)
        if num >= max_frame:
            num = 0
            cutList.append(i)
            i += min_frame
        elif res < 95:
            res2 = SSIM(img0, img1)
            if res2 < 60:
                res3 = (95 - res) ** 2 + (60 - res2) ** 2
                if (res3 > 150):
                    num = 0
                    cutList.append(i)
                    i += min_frame
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return cutList


def saveToDir(images, cutList, dirName):
    current_file_path = __file__
    absolute_path = os.path.abspath(current_file_path)
    directory = os.path.dirname(absolute_path)

    all_cut_dir = os.path.join(directory, "VideoCutDir")
    if not os.path.exists(all_cut_dir):
        os.mkdir(all_cut_dir)
    root_dir = os.path.join(all_cut_dir, dirName)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.mkdir(root_dir)
    cut_dir = os.path.join(root_dir, dirName + str(0).zfill(3))
    os.mkdir(cut_dir)
    out_dir = cut_dir + '\n'
    for i in range(len(images)):
        image = tensor_to_cv2(images[i])
        image_path = os.path.join(cut_dir, str(i).zfill(6) + ".png")
        cv2.imwrite(image_path, image)
        if i in cutList:
            num = len(os.listdir(root_dir))
            cut_dir = os.path.join(root_dir, dirName + str(num).zfill(3))
            os.mkdir(cut_dir)
            out_dir += cut_dir + '\n'
    return out_dir
