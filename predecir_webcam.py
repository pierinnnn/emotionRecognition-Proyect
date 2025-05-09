import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

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


        emotion_counts[emotion] += 1

        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detección de emociones', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


def guardar_reporte_pdf(emotion_counts, filename="reporte_emociones.pdf"):

    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []


    titulo = Paragraph("<b>Reporte de emociones detectadas</b>", styles['Title'])
    story.append(titulo)
    story.append(Spacer(1, 12))


    intro = Paragraph("Este reporte muestra un resumen de las emociones detectadas durante la ejecución del programa.",
                      styles['BodyText'])
    story.append(intro)
    story.append(Spacer(1, 12))


    data = [["Emoción", "Detecciones"]]
    for emotion, count in emotion_counts.items():
        data.append([emotion, str(count)])

    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 24))


    labels = list(emotion_counts.keys())
    sizes = list(emotion_counts.values())
    colors_pie = ['#FF9999', '#66B2FF', '#99FF99', '#66B2FF', '#99FF99', '#66B2FF', '#99FF99']
    plt.figure(figsize=(6, 4))
    plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribución de emociones detectadas')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    pie_chart = Image(buf, width=400, height=300)
    story.append(pie_chart)
    story.append(Spacer(1, 12))


    footer = Paragraph("<i>Reporte generado automáticamente por tu sistema de detección de emociones</i>",
                       styles['Italic'])
    story.append(footer)

    doc.build(story)
    print(f"Reporte guardado en {filename}")

guardar_reporte_pdf(emotion_counts)
