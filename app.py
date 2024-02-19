from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, url_for, Response
import cv2
import datetime
from twilio.rest import Client
from werkzeug.utils import redirect


app = Flask(__name__)

observations = []


account_sid = 'AC86a224f852c64d861eb5e81e1f84f607'
auth_token = 'c8e59d00ba328f35f461ccc690197c5c'
client = Client(account_sid, auth_token)


class Emotion:
    def __init__(self, emotion):
        timestamp = datetime.datetime.now()
        self.emotion = emotion
        self.date_time = timestamp.strftime("% Y-%m-%d % H: % M: % S")

    # model = load_model('bin_aut.h5')
model = load_model('Mymodel.h5')
# model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


def gen_frames():
    i = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')
            # eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                if(i % 100 == 0):
                    img = roi_color
                    # img = cv2.imread('')
                    img = cv2.resize(img, (150, 150))
                    img = np.reshape(img, [1, 150, 150, 3])
                    # classes = model.predict(img)
                    classes = (model.predict(img) > 0.5).astype("int32")
                    global emotion
                    if classes[0][2] == 1:
                        emotion = Emotion('Neutral')
                    else:
                        emotion = Emotion('Distress')
                        message = client.messages.create(
                            messaging_service_sid='MGe4043595637b19189f95b497ebf0c7b2',
                            body='Your kid is not feeling well',
                            to='+919400775095'
                        )
                        print(message.sid)

                    observations.append(emotion)

                    print(classes)

                i = i+1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/webcam')
def webcam():
    global camera
    camera = cv2.VideoCapture(0)
    return render_template('webcam.html', emotions=observations)


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop-video')
def stop():
    camera.release()
    return redirect('/')


@app.route('/')
def index():
    return render_template('homepage.html')


if __name__ == "__main__":
    app.run(debug=True)
