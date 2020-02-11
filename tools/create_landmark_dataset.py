import dlib
import os
import sys
import cv2
import csv
from skimage import io
import numpy as np

predictor_path = "./tools/shape_predictor_68_face_landmarks.dat"
faces_dir = './tools/data/faces/'
face_landmarked_dir = './tools/data/face_landmarks/'
csv_path = './tools/data/face_landmarks.csv'

def face_landmarks(dlib_model_path, faces_image_path, csv_path):
    if dlib_model_path is None:
        raise ValueError('dlib_model_path is None')
    
    print('face_landmark is startting...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_model_path)

    print('face_landmark is running...')

    #创建csv文件
    csv_file = open(csv_path, 'w', newline="")
    if csv_file is None:
        raise ValueError('create csv file failed!')
    csv_writer = csv.writer(csv_file)

    #写入csv列名
    csv_columname = ['image_name']
    for i in range(68):
        colName = 'part{}'.format(i)
        csv_columname.append(colName)

    csv_writer.writerow(csv_columname)

    #遍历faces_dir的所有图片
    for root, dirs, files in os.walk(faces_image_path):
        for file in files:
            if (1 - file.endswith('.jpg')) and (1 - file.endswith('.png')):
                continue

            image_file = os.path.join(root, file)
            print('Read File:{}'.format(image_file))
            img = io.imread(image_file)
            dets = detector(img, 1)

            for k, d in enumerate(dets):
                shape = predictor(img, d)
                landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
                points = []
                for idx, point in enumerate(landmark):
                    pos = (point[0, 0], point[0, 1])
                    cv2.circle(img, pos, 3, color=(0, 255, 0))
                    points.append(pos[0])
                    points.append(pos[1])

                #数据写入csv文件，格式：文件名 x0 y0 x1 y1 x2 y2 ...... x67 y67
                faceinfo = [file]
                for idx, pt in enumerate(points):
                    faceinfo.append(pt)

                csv_writer.writerow(faceinfo)

            face_landmarked_path = os.path.join(face_landmarked_dir, file)
            io.imsave(face_landmarked_path, img)

    csv_file.close()
    print('face_landmark is Done!')


if __name__ == "__main__":
    face_landmarks(predictor_path, faces_dir, csv_path)