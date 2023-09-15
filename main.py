import os
import cv2
import face_recognition
import numpy as np
import csv

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        images.append(image)
    return images

def get_encodings(images):
    face_encodings = []
    for image in images:
        face_encoding = face_recognition.face_encodings(image)[0]
        face_encodings.append(face_encoding)
    return face_encodings

def mark_attendance(attendance_csv, name):
    with open(attendance_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, 'Present'])

def main():
    known_faces_folder = "known_faces"
    attendance_csv = "attendance.csv"

    known_faces = []
    known_names = []

    for folder_name in os.listdir(known_faces_folder):
        person_folder = os.path.join(known_faces_folder, folder_name)
        images = load_images_from_folder(person_folder)
        face_encodings = get_encodings(images)
        known_faces.extend(face_encodings)
        known_names.extend([folder_name] * len(face_encodings))

    video_capture = cv2.VideoCapture(1)
    recognized_names = set()

    while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            name = "Unknown"

            if True in matches:
                matched_indices = [i for i, val in enumerate(matches) if val]
                name = known_names[min(matched_indices)]

            y1, x2, y2, x1 = face_locations[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)

            # Mark attendance if recognized face and not already marked
            if name != "Unknown" and name not in recognized_names:
                mark_attendance(attendance_csv, name)
                recognized_names.add(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
