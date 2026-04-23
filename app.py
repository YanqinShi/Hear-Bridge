from flask import Flask, request, jsonify
import speech_recognition as sr
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

@app.route('/api/speech-to-sign', methods=['POST'])
def speech_to_sign():
    data = request.get_json()
    text = data['text']
    # 使用某种方法将文本转换为手语动画
    # 返回手语动画的URL或数据
    return jsonify({'animation': 'URL_to_animation'})

@app.route('/api/sign-recognition', methods=['POST'])
def sign_recognition():
    data = request.get_json()
    image_data = data['image']

    # 将base64编码的图像数据转换为OpenCV图像
    image_data = image_data.split(',')[1]  # 去掉data:image/png;base64,前缀
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 处理图像数据并进行手语识别
    # 这里可以添加手语识别的代码
    # 返回识别的文本
    recognized_text = 'Recognized Text'
    
    return jsonify({'text': recognized_text})

if __name__ == '__main__':
    app.run(debug=True)
