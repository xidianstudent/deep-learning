import os
from skimage import io
import cv2
import csv

faces_dir = './tools/data/faces/'
face_landmarked_dir = './tools/data/face_landmarks/'
csv_file = './tools/data/face_landmarks.csv'

def draw_landmark_and_save(face_path, landmarked_path, csv_path):
    if  not os.path.exists(csv_path):
        raise ValueError('csv file is not exist!')

    print('draw_landmark_and_save is starting...')
    csv_file = open(csv_path, 'r')
    csv_reader = csv.reader(csv_file)

    next(csv_reader) #跳过第一行（列名)
    for row in csv_reader:
        # print(type(row))
        fileName = row[0]
        print('Processing File:{}'.format(fileName))
        filePath = os.path.join(face_path, fileName)
        img = io.imread(filePath)
        for i in range(1, len(row), 2):
            pos = (int(row[i]), int(row[i + 1]))
            cv2.circle(img, pos, 3, color=(0, 255, 0))
        
        saved_path = os.path.join(landmarked_path, fileName)
        io.imsave(saved_path, img)
    
    csv_file.close()
    print('draw_landmark_and_save is Done...')


if __name__ == "__main__":
    draw_landmark_and_save(faces_dir, face_landmarked_dir, csv_file)