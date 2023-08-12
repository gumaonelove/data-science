import os
import pickle

import cv2
from PIL import Image
from deepface import DeepFace
from transliterate import translit


def xywh2xyxy(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


class DeepFaceHandler:
    def __init__(self, faces_path: str, detector: str = 'mediapipe', extractor: str = 'VGG-Face'):
        self.faces_path = faces_path
        self.detector = detector
        self.extractor = extractor

    def form_path(self, name_employee: str):
        eng_name_employee = translit(name_employee, language_code='ru', reversed=True)
        eng_name_employee = '_'.join(eng_name_employee.split())
        new_path = self.faces_path + '/' + eng_name_employee + '.jpg'
        return new_path

    def save_face(self, path_to_image: str, name_employee: str, model: str = 'vgg_face'):
        '''
        :param path: path to faces folder
        :param path_to_image: full path to new image
        :param name_employee: name of new employee in format: Имя Фамилия
        :return:
        '''
        img = Image.open(path_to_image)
        new_path = self.form_path(name_employee)
        img.save(new_path)
        embs = []
        try:
            with open(fr"{self.faces_path}\representations_{model}.pkl", 'rb') as file:
                embs = pickle.load(file)
        except:
            pass

        new_face_embedding = DeepFace.represent(img_path=path_to_image, model_name=self.extractor,
                                                detector_backend=self.detector)
        if [new_path, new_face_embedding[0]['embedding']] not in embs:
            embs.append([new_path, new_face_embedding[0]['embedding']])

        with open(fr"{self.faces_path}\representations_{model}.pkl", 'wb') as file:
            pickle.dump(embs, file)

    def delete_face(self, name_employee: str, model: str = 'vgg_face'):
        path_to_delete = self.form_path(name_employee)
        if path_to_delet[path_to_delet.rfind('/') + 1:] in os.listdir(self.faces_path):
            os.remove(path_to_delet)
            with open(fr"{self.faces_path}\representations_{model}.pkl", 'rb') as file:
                embs = pickle.load(file)
                for emb in embs:
                    if emb[0] == path_to_delete:
                        embs.remove(emb)
            print('Данные сотрудника удалены')
        else:
            print('Сотрудника нет в базе лиц')

    def main_loop(self):
        cap = cv2.VideoCapture(0)
        vgg_treshold_cosine = 0.16
        # arcface_trashold_cosine = 5
        # num_frames = 0
        # first_employee_photo = self.faces_path + "/" + os.listdir(self.faces_path)[1]
        # print(first_employee_photo)
        # DeepFace.find(first_employee_photo, detector_backend=self.detector, model_name=self.extractor, db_path=self.faces_path)
        while True:
            ret, frame = cap.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # try:
            # one or several persons?
            # try:

            df = DeepFace.find(frame, detector_backend=self.detector, model_name=self.extractor,
                               db_path=self.faces_path,
                               enforce_detection=False, silent=True)
            if not df[0]['identity'].empty:
                face_path = df[0]['identity'][0]
                face_name = translit(face_path[face_path.rfind("/") + 1:-4], language_code="ru").split("_")
                if df[0]['VGG-Face_cosine'][0] < vgg_treshold_cosine:
                    # for situations like this: Роман леонтЬев
                    face_name = " ".join([fn[0].upper() + fn[1:].lower() for fn in face_name])
                    print(f'Распознан {face_name}')
                    print(df[0])
                    # num_frames += 1

                    left, top, right, bottom = xywh2xyxy(df[0]['source_x'][0],
                                                         df[0]['source_y'][0],
                                                         df[0]['source_w'][0],
                                                         df[0]['source_h'][0])
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, translit(face_name, language_code="ru", reversed=True), (left + 6, bottom + 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.003 * (bottom - top),
                                (255, 255, 255), 1)
            # except:
            # print('Лицо не распознано')
            # print(df[0])

            # except:
            # print('Лицо не распознано')

            cv2.imshow('image', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


path = r"C:\Users\user\Downloads\faces"
dfh = DeepFaceHandler(path, detector='mediapipe')

dfh.main_loop()
