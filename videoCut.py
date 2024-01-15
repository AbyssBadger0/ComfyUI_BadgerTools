import os
import subprocess
import shutil
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import sys
from skimage import metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)


def SSIM(imgPath0, imgPath1):
    image1 = cv2.imread(imgPath0)
    image2 = cv2.imread(imgPath1)
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
    return round(ssim_score[0], 2) * 100


def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score


def getCutList(imagePath, min_frame, max_frame):
    pngList = os.listdir(imagePath)
    cutList = []
    resList = []
    indexList = []
    i = 0
    num = 0
    while i < len(pngList) - 1:
        num += 1
        imgPath0 = os.path.join(imagePath, pngList[i])
        imgPath1 = os.path.join(imagePath, pngList[i + 1])
        res = generateScore(imgPath0, imgPath1)
        print("切割画面（" + str(i + 1) + "/" + str(len(pngList) - 1) + ")  相似度：" + str(res) + "%")
        resList.append(res)
        indexList.append(i)
        if num >= max_frame:
            num = 0
            cutList.append(pngList[i])
            i += min_frame
        elif res < 95:
            res2 = SSIM(imgPath0, imgPath1)
            if res2 < 60:
                res3 = (95 - res) ** 2 + (60 - res2) ** 2
                if (res3 > 150):
                    num = 0
                    cutList.append(pngList[i])
                    i += min_frame
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    return cutList


def videoToPng(videopath, rate, save_name):
    current_file_path = __file__
    absolute_path = os.path.abspath(current_file_path)
    directory = os.path.dirname(absolute_path)

    all_cut_dir = os.path.join(directory, "VideoCutDir")
    if not os.path.exists(all_cut_dir):
        os.mkdir(all_cut_dir)
    root_dir = os.path.join(all_cut_dir, save_name)
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.mkdir(root_dir)
    ffmpegCMD = [
        'ffmpeg',
        '-i', videopath,  # 输入视频路径
        '-vf', f'fps={rate}',  # 设置帧率
        '-q:v', '2',  # 设置输出质量
        os.path.join(root_dir, 'frame_%05d.png')  # 输出文件名和格式
    ]
    subprocess.run(ffmpegCMD)
    print("视频转图片完成")

    return root_dir


def cutToDir(root_dir, cutList):
    dirIndex = 0
    pngList = os.listdir(root_dir)
    dirList = []
    dirPathString = ""
    for i in range(len(cutList) + 1):
        dirName = str(i).zfill(3)
        dirPath = os.path.join(root_dir, dirName)
        os.mkdir(dirPath)
        dirPathString += (dirPath + '\n')
        dirList.append(dirName)
    for png in pngList:
        src = os.path.join(root_dir, png)
        tgtDir = os.path.join(root_dir, dirList[dirIndex])
        shutil.move(src, tgtDir)
        if png in cutList:
            dirIndex += 1

    return dirPathString
