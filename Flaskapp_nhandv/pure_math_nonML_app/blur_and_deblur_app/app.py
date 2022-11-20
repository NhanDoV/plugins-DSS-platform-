import cv2, math, os
import numpy as np
from numpy import fft
from flask import Flask, render_template, request

def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1

        return PSF / PSF.sum()  # Normalize the luminance of the point spread function

    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1

        return PSF / PSF.sum()

def wiener(input, PSF, eps, K=0.01):  # Wiener filtering，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))

    return result

def deblur(image, eps, K):
    img_h, img_w = image.shape[:2]
    PSF = motion_process((img_h, img_w), 10)
    result = wiener(image, PSF, eps)

    return result

def bluring(image, blur_type, ksize, diam = 9, sigma_color = 11, sigma_space = 25):
    if blur_type == '2D_filtering':
        blur_img = cv2.blur(image, (ksize, ksize))
    elif blur_type == 'Gaussian':
        blur_img = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == 'Median':
        blur_img = cv2.medianBlur(image, ksize)
    elif blur_type == 'Bilateral':
        blur_img = cv2.bilateralFilter(image, diam, sigma_color, sigma_space)
    return blur_img

def transformed(transformed_type, image, blur_type, ksize, P3, P4, P5, P6, diam = 9, sigma_color = 11, sigma_space = 25, eps=1e-3, K=1e-2):
    ksize = int(ksize)
    diam = int(diam)
    sigma_color = int(sigma_color)
    sigma_space = int(sigma_space)
    eps = float(eps)
    K = float(K)
    P3 = int(P3)
    P4 = int(P4)
    P5 = int(P5)
    P6 = int(P6)

    if transformed_type == 'bluring':
        res_img = bluring(image, blur_type, ksize, diam, sigma_color, sigma_space)
    elif transformed_type == 'debluring':
        res_img = deblur(image, eps, K)
    else:
        res_img = cv2.fastNlMeansDenoisingColored(image, None, P3, P4, P5, P6)

    return res_img

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/' , methods = ['GET', 'POST'])
def main():

    if request.method == 'POST':

        # Load params
        transformed_type = request.form['transformed_type']
        blur_type = request.form['blur_type']
        img_file = request.files['file']
        ksize = request.form['ksize']
        diam = request.form['diam']
        sigma_color = request.form['sigma_color']
        sigma_space = request.form['sigma_space']
        eps = request.form['eps']
        K = request.form['Kw']
        P3 = request.form['P3']
        P4 = request.form['P4']
        P5 = request.form['P5']
        P6 = request.form['P6']

        img_name = os.path.join('static', 'input.png')
        img_file.save(os.path.join('static', 'input.png'))

        if transformed_type == 'debluring':
            image = cv2.imread(img_name, 0)
        else: 
            image = cv2.imread(img_name)

        pred_img = transformed(transformed_type, image, blur_type, ksize, P3, P4, P5, P6, diam, sigma_color, sigma_space, eps, K)
        output_fname = os.path.join('static', 'output.png')

        cv2.imwrite(output_fname, pred_img)

        return render_template('home.html', 
                                uploaded_image = img_name,
                                transformed_img = output_fname,
                                output_type = transformed_type
                            )

if __name__ == "__main__":
    app.run(port='8080', threaded=False, debug=True)