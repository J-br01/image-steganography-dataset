import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math


def import_image(image):
    img = Image.open(image)
    data = np.asarray(img)
    img_bytes = img.tobytes()
    return data.shape, data, len(img_bytes)


def check_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        return False
    else:
        return True


def metrics(cover, stego, msg):
    data_c, array_c, bytes_c = import_image(cover)
    data_s, array_s, bytes_s = import_image(stego)
    if check_file_extension(msg) == True:
        data_h, array_h, bytes_h = import_image(msg)
    else:
        bytes_h = os.path.getsize(msg)
    print('-----------------------------------------------')
    print('COVER: ' + os.path.basename(cover))
    print('STEGO: ' + os.path.basename(stego))
    print('MESSAGE: ' + os.path.basename(msg))
    # Embedding Error calcualtion (ER bpp)
    h_w = data_c[0] * data_c[1]
    bits = bytes_h * 8
    er = bits / h_w
    print('ER: ', er)
    # MSE calculation
    if len(data_c) != len(data_s):
        print('Array length is different')
    else:
        squared_diff = (array_c - array_s) ** 2
        mse = np.mean(squared_diff)
        print('MSE: ', mse)

        # PSNR calculation
        sqr = math.sqrt(mse)
        div = (255) / sqr
        log = math.log10(div)
        psnr = 20 * log
        print('PSNR: ', psnr)
        return er, mse, psnr


def image_histogram(image):
    img = Image.open(image)
    array_img = np.array(img)
    if len(array_img.shape) == 2:
        histogram_general = np.histogram(array_img, bins=256, range=(0, 256))[0]
        prob = histogram_general / (array_img.shape[0] * array_img.shape[1])
    else:
        red = np.histogram(array_img[:, :, 0], bins=256, range=(0, 256))[0]
        green = np.histogram(array_img[:, :, 1], bins=256, range=(0, 256))[0]
        blue = np.histogram(array_img[:, :, 2], bins=256, range=(0, 256))[0]
        histogram_general = red + blue + green
        prob = histogram_general / (array_img.shape[0] * array_img.shape[1] * 3)

    return prob, histogram_general


def entropy_gen(path):
    for filename in os.listdir(path):
        if filename == '.DS_Store':
            continue
        file_path = os.path.join(path, filename)
        f1 = os.path.basename(file_path).replace(".jpg", "")
        prob, general = image_histogram(file_path)
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        print(f'{f1} entropy\'s: {entropy}')


def histogram_plot(histogram, file):
    plt.figure()
    plt.title(f'Histogram for {file}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.bar(np.arange(256), histogram, color='black', alpha=0.6)
    plt.savefig(f'/Users/judybdma/Desktop/Steganography/histograms/jstego/{file}')
    plt.show()


if __name__ == '__main__':
    path = ' ' #Path to stego image folder
    path2 = '' #Path to cover image folder
    hidden_msg = ' ' #path to hidden message file
    total_er = []
    total_psnr = []
    total_mse = []
    total = []
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        total.append(1)
        file_path = os.path.join(path, file)
        for file2 in os.listdir(path2):
            file_path2 = os.path.join(path2, file2)
            f1 = os.path.basename(file_path).replace(".jpg", "")
            f2 = os.path.basename(file_path2).replace(".jpg", "")
            if f2 in f1:
                metrics(file_path2, file_path, hidden_msg)


