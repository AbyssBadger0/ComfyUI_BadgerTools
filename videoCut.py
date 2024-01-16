import os
import subprocess
import shutil
import cv2
import skimage


def calculate_image_similarity(img_path1, img_path2):
    # 读取图片
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # 检查图片是否有效
    if img1 is None or img2 is None:
        raise ValueError("One of the images couldn't be loaded.")

    # 缩放图片到相同大小
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 方法1: 直方图比较
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # 方法2: 结构相似性指数 (SSIM)
    ssim_similarity = skimage.metrics.structural_similarity(img1, img2)

    # 方法3: 特征点匹配 (使用ORB)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    feature_similarity = len(matches) / float(min(len(des1), len(des2)))
    # 综合评分
    combined_score = (hist_similarity + ssim_similarity + feature_similarity) / 3.0

    return combined_score


def getCutList(imagePath, threshold, min_frame, max_frame):
    pngList = os.listdir(imagePath)
    cutList = []
    indexList = []
    resList = []
    i = min_frame - 1
    num = 0
    while i < len(pngList) - 1:
        num += 1
        imgPath0 = os.path.join(imagePath, pngList[i])
        imgPath1 = os.path.join(imagePath, pngList[i + 1])
        similarity = calculate_image_similarity(imgPath0, imgPath1)
        print("切割画面(" + str(i + 1) + "/" + str(len(pngList) - 1) + ") 相似度:" + str(similarity))
        indexList.append(i)
        resList.append(similarity)
        i += 1

    i = min_frame - 1
    avg = sum(resList) / len(resList)
    threshold = avg * threshold
    while i < len(resList) - 1:
        if num >= max_frame:
            num = 0
            cutList.append(pngList[i])
            i += min_frame
        elif resList[i] < threshold:
            num = 0
            cutList.append(pngList[i])
            i += min_frame
        else:
            i += 1
    print("平均计算阈值："+str(avg)+"   处理阈值："+str(threshold))
    print(cutList)
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


def frames_to_video(input_folder, frame_rate, output_path):
    # Get a sorted list of (non-hidden) image files
    files = [f for f in sorted(os.listdir(input_folder)) if not f.startswith('.')]
    sorted_files = sorted(files, key=lambda x: os.path.splitext(x)[0])

    # Create a temporary file listing all images for ffmpeg
    temp_list_path = 'ffmpeg_temp_list.txt'
    with open(temp_list_path, 'w') as filelist:
        for filename in sorted_files[:-1]:
            file_path = os.path.join(input_folder, filename)
            filelist.write(f"file '{file_path}'\n")
            filelist.write(f"duration {1 / frame_rate}\n")
        # Write the last file without a duration
        filelist.write(f"file '{os.path.join(input_folder, sorted_files[-1])}'\n")

    # Construct the ffmpeg command to create the video
    command = [
        'ffmpeg',
        '-f', 'concat',  # Use the concat demuxer
        '-safe', '0',  # Allow unsafe file paths
        '-i', temp_list_path,  # Input file list
        '-vsync', 'vfr',  # Variable frame rate mode to match input sequence
        '-pix_fmt', 'yuv420p',  # Pixel format, widely compatible
        '-c:v', 'libx264',  # Codec to use for encoding
        '-crf', '23',  # Constant Rate Factor, controls quality (lower is better)
        output_path
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f'Video has been created at {output_path}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred: {e}')
    finally:
        # Clean up temporary file
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)


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
