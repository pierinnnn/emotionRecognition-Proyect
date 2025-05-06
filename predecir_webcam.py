import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

model = load_model('modelo_emociones.h5')

img_size = (48, 48)
batch_size = 32
train_dir = '../EmocionesData/train'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical'
)
class_labels = list(generator.class_indices.keys())

emotion_counts = {emotion: 0 for emotion in class_labels}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        max_index = np.argmax(prediction[0])
        emotion = class_labels[max_index]

        # Actualizar contador
        emotion_counts[emotion] += 1

        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detección de emociones', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


def guardar_reporte_pdf(emotion_counts, filename="reporte_emociones.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    text = c.beginText(40, height - 40)
    text.setFont("Helvetica", 12)
    text.textLine("Reporte de emociones detectadas")
    text.textLine("")
    for emotion, count in emotion_counts.items():
        text.textLine(f"{emotion}: {count} detecciones")
    c.drawText(text)
    c.save()
    print(f"Reporte guardado en {filename}")


guardar_reporte_pdf(emotion_counts)