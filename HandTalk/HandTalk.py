import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import os

class HandGestureDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.model = load_model("C:\\Users\\Gusta\\OneDrive\\Documentos\\GitHub\\HandTalk\\HandTalk\\keras_model.h5")
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'Eu te amo']
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def detect_hand_region(self, hand_landmarks, image_shape):
        x_max, y_max = 0, 0
        x_min, y_min = image_shape[1], image_shape[0]

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * image_shape[1]), int(lm.y * image_shape[0])
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            x_min = min(x_min, x)
            y_min = min(y_min, y)

        return x_min, y_min, x_max, y_max

    def preprocess_image(self, img, x_min, y_min, x_max, y_max):
        img_crop = img[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
        img_crop = cv2.resize(img_crop, (224, 224))
        img_array = np.asarray(img_crop)
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

        self.data[0] = normalized_image_array
        return self.data

class DataCollector:
    def __init__(self):
        self.data_dir = 'collected_data'
        self.counter = 0

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def collect_data(self, image, hand_position, predicted_class, confidence):
        filename = os.path.join(self.data_dir, f'data_{self.counter + 1}.npz')
        self.counter += 1

        np.savez(filename,
                image=image,
                hand_position=hand_position,
                predicted_class=predicted_class,
                confidence=confidence)

class GestureRecognitionApp:
    def __init__(self):
        self.hand_detector = HandGestureDetector()
        self.data_collector = DataCollector()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Largura
        self.cap.set(4, 480)  # Altura

    def run(self):
        while True:
            success, img = self.cap.read()
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.hand_detector.hands.process(frame_rgb)
            hands_points = results.multi_hand_landmarks

            if hands_points:
                for hand in hands_points:
                    x_min, y_min, x_max, y_max = self.hand_detector.detect_hand_region(hand, frame_rgb.shape)

                    try:
                        img_crop = self.hand_detector.preprocess_image(frame_rgb, x_min, y_min, x_max, y_max)
                        prediction = self.hand_detector.model.predict(img_crop)

                        index_val = np.argmax(prediction)
                        confidence = prediction[0][index_val]

                        if confidence > 0.8:
                            self.data_collector.collect_data(img_crop, (x_min, y_min, x_max, y_max),
                                self.hand_detector.classes[index_val], confidence)

                            cv2.putText(img, self.hand_detector.classes[index_val], (x_min - 50, y_min - 65),
                                        cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

                    except Exception as e:
                        print(f"Erro durante a predição: {str(e)}")

            cv2.imshow('Imagem', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureRecognitionApp()
    app.run()