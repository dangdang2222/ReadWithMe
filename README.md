### ReadWithMe
## Gaze Estimation &amp; OCR &amp; ChatGPT API &amp; Chrome Extension
Sogang University / Capstone Design I / CSE4186

## Project Overview

The goal is to develop and distribute a browser extension that enhances user browsing experience by combining Gaze Estimation, Optical Character Recognition (OCR), and the ChatGPT API. This extension will assist users in navigating web pages easily, recognizing text, and obtaining additional information and services through conversational capabilities.\

## Pipeline
![Untitled](https://github.com/dangdang2222/ReadWithMe/assets/64007947/8d222631-4800-406d-8c56-0da12a01f0be)

The project consists of three main parts: **Gaze Estimation**, **OCR** and the **ChatGPT API**, all combined within a **Chrome extension**. These parts were distributed and developed in parallel.

The **Gaze Estimation** model receives webcam images from the Chrome extension and tracks the user's gaze in real-time. It uses the **L2CS-Net** model, modifying the existing model to estimate the coordinates of the point the user is looking at on the screen. To refine this estimation, a heatmap using Gaussian blobs is utilized to determine the user's focal point.

For **OCR**, the **Tesseract** model is employed. It extracts words located at the specified coordinates on the captured screen, using the information of the current displayed image and its corresponding coordinates. An algorithm is then implemented to extract sentences containing these words. To communicate with **ChatGPT**, the OpenAI module is utilized. This enables the Python server to receive answers from the ChatGPT API, based on the extracted sentences and words, when the user asks relevant questions.

The **Chrome extension** sends webcam images to the Gaze Estimation model and receives the resulting coordinate values. It also sends the captured full-screen image, along with these coordinates, to the OCR module to extract text. Finally, the ChatGPT question-answer results are obtained and displayed to the user.

## Setting

1.  Go to the below link to L2CS-Net and follow the step for Installation and the environment setting.

https://github.com/Ahmednull/L2CS-Net

2. Clone this repository

```python
git clone https://github.com/dangdang2222/ReadWithMe.git
```

All files, except for the "extension_demo/extension" folder, should be placed within the cloned directory of L2CS.

3. Installation

(수정 필요)

```python
pip install ??
```

4. For the gaze estimation model demo use below code

```python
python demo06014.py --snapshot models/L2CSNet_gaze360.pkl --gpu 0 --cam 0
```

5. Overall pipeline demo

Check our Demo Video
![캡디 데모 - Clipchamp로 제작 (2)](https://github.com/dangdang2222/ReadWithMe/assets/64007947/dfb18302-f672-4a97-a7aa-87d28562c801)
