import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)  # ou flux vidéo

face_exp_model = model_from_json(
    open("dataset/facial_expression_model_structure.json", "r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_label = ('pas content', 'vomi', 'chocottes',
                  'tristounet', 'graou', 'surprise', 'ryan gosling')

all_face_locations = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=1, model="hog")
    all_face_encodings = face_recognition.face_encodings(
        current_frame_small, all_face_locations)

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        current_face_image = current_frame[top_pos:bottom_pos,
                                           left_pos:right_pos]
        current_face_image = cv2.cvtColor(
            current_face_image, cv2.COLOR_BGR2GRAY)
        current_face_image = cv2.resize(current_face_image, (48, 48))
        img_pixels = image.img_to_array(current_face_image)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        exp_predictions = face_exp_model.predict(img_pixels)
        max_index = np.argmax(exp_predictions[0])
        emotion_label = emotions_label[max_index]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, emotion_label, (left_pos,
                    bottom_pos), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Webcam Video", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

webcam_video_stream.release()
cv2.destroyAllWindows()
