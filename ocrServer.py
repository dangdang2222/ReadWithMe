from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sse import sse

from io import BytesIO
import base64
import re
import json
import PIL
from PIL import Image
import numpy as np

import cv2
import pytesseract
from pytesseract import Output

import time
import datetime

import openai


app = Flask(__name__)
CORS(app)
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream')

pytesseract.pytesseract.tesseract_cmd = R'C:\Program Files\Tesseract-OCR\tesseract.exe'

def distance(x, y, w, h, X, Y):

    return pow(X - (x+w/2), 2.0) + pow(Y - (y+h/2), 2.0)


def update_best_sentence(sentence, sentence_list):


    words = sentence.split(" ")
    (first, mid, last) = (words[0], words[len(words)//2], words[-1])


    count = 0

    for i in range(len(sentence_list)):
        words_ = sentence_list[i].split(" ")

        (first_, mid_, last_) = (words_[0], words_[len(words_)//2], words_[-1])

        if first == first_ and mid == mid_ and last == last_:
            count += 1

    return count


@app.route("/extract", methods=["POST"])
def test():
    image_data = re.sub('^data:image/.+;base64,', '', request.form['image'])
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((1600,1200))
    list = []
    word_index_list = []
    sentence_list = []
    sentence_dup_num = []
    
    subList = request.form["coordinates"]
    subList = eval(subList)
    
    list = [(int(x), int(y)) for x, y in subList]
    
    #X = int(request.form['x'])
    #Y = int(request.form['y'])
  
    text = ""  # 단어 하나하나
    

    sentence = ""
    line1 = ""
    line2 = ""

    word_gap = 0
    line_gap = 0
    
    
    #print("X : ", X, " , Y : " , Y)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    
    
    n_boxes = len(d['text'])
    
    for index in range(len(list)):
        (X, Y) = list[index]

        dis = float('inf')
        dis_min_index = -1
        for i in range(n_boxes):
            (x, y, w, h, wd) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['text'][i])

            if len(wd) == 0:
                continue

            dis_ = distance(x, y, w, h, X, Y)

            if dis > dis_:
                dis = dis_
                dis_min_index = i
    
        word_index_list.append(dis_min_index)
        

#print(word_index_list)

    for i in range(len(word_index_list)):

        idx = word_index_list[i]

        (x, y, w, h) = (d['left'][idx], d['top'][idx], d['width'][idx], d['height'][idx])



        current_block = d['block_num'][idx]

        a = idx-1

        while True:

            if a < 0:
                break

            (x, y, w, h, b_num, word) = (d['left'][a], d['top'][a], d['width'][a], d['height'][a], d['block_num'][a], d['text'][a])

        #print_component(d, a)

            if "." in d['text'][a] or b_num != current_block:
                break
        

            if len(word) != 0:
                sentence = word + " " + sentence
    
            a -= 1
    
        b = idx + 1

        while True:

            if b > n_boxes:
                break

        #print_component(d, b)
            (x, y, w, h, b_num, word) = (d['left'][b], d['top'][b], d['width'][b], d['height'][b], d['block_num'][b], d['text'][b])
        
            if  b_num != current_block:
                break

            if  "." in d['text'][b]:
                sentence = sentence + " " + word
                break

            if len(word) != 0:
                sentence = sentence + " " + word
            b += 1

    
    #print(sentence)

        sentence_dup_num.append(update_best_sentence(sentence, sentence_list))
        sentence_list.append(sentence)
        sentence = ""
                    

    end = time.time()
    max_dup_value = -1
    max_dup_idx = -1
    
    for i in range(len(sentence_dup_num)):

        if sentence_dup_num[i] > max_dup_value:
            max_dup_value = sentence_dup_num[i]
            max_dup_idx = i
    
    print("sentence : ", sentence_list[max_dup_idx])

   # return jsonify({"answer": response.choices[0].text.strip()})

    
    return jsonify({"answer":sentence_list[max_dup_idx]})
    
@app.route("/answer", methods=["POST"])
def chatGPT():
    sentence = request.form['sentence']
    openai.api_key = "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"
    
    prompt = ""
    prompt += "\"" + sentence + "\""
    prompt += "에 대해서 한국어로 설명하고 추가적으로 더 자세하게 알려줘"
    
    
    responses = openai.Completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        temperature = 0.3,
        max_tokens = 1000,
        top_p = 1,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        best_of=1,
        stream = True
    )

    for res in responses:
        print("response: ", res.choices[0].text.strip())
        sse.publish({"message": res.choices[0].text.strip()}, type='gpt_response')
  

    
    return jsonify({"answer": "Check SSE for answer"})
    

if __name__ == "__main__":
    app.run(debug=True,port=8000)
