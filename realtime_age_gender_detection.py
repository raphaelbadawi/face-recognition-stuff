import cv2
import face_recognition

webcam_video_stream = cv2.VideoCapture(0)  # ou flux vidéo

all_face_locations = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=1, model="hog")

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        current_face_image = current_frame[top_pos:bottom_pos,
                                           left_pos:right_pos]
        # values calculated using numpy.mean(blob.channels, blob.height, blob.width) after converting mean.binaryproto to blob
        AGE_GENDER_MODEL_MEAN_VALUES = (
            78.4263377603, 87.7689143744, 114.895847746)
        current_face_image_blob = cv2.dnn.blobFromImage(
            current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)

        gender_label_list = ["Male", "Femelle"]
        gender_protext = "dataset/gender_deploy.prototxt"
        gender_caffemodel = "dataset/gender_net.caffemodel"
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]

        age_label_list = ["bébé", "presque bébé", "enfant", "chadolescent",
                          "jeune adulte", "adulte", "vieux", "très vieux schnoque"]
        age_protext = "dataset/age_deploy.prototxt"
        age_caffemodel = "dataset/age_net.caffemodel"
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(current_face_image_blob)
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]

        cv2.rectangle(current_frame, (left_pos, top_pos),
                      (right_pos, bottom_pos), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, age + " " + gender, (left_pos,
                    bottom_pos), font, 0.5, (0, 255, 0), 1)

        cv2.imshow("Webcam Video", current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

webcam_video_stream.release()
cv2.destroyAllWindows()
