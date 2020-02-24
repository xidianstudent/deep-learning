import cv2
import dlib
import numpy as np
import os
import time

class Config(object):
    predictor_path = './tools/shape_predictor_68_face_landmarks.dat'
    test_img = 'test.jpg'
    width = 640
    height = 480


class FaceDetective():

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(Config.predictor_path)

    def check_file(self,path):
        if os.path.exists(path):
            img = cv2.imread(path)
            return img
        else:
            raise IOError('No such file : "%s", please check!' % path)

    def detective(self, frame):
        starttime = time.time() * 1000
        faces = self.detector(frame, 0)
        if len(faces):
            print ('==> Found %d face in this frame.'%(len(faces)))
            for i in range(len(faces)):
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, faces[i]).parts()])
                for point in landmarks:
                    pos = (point[0, 0], point[0, 1])
                    cv2.circle(frame, pos, 3, color=(0, 0, 255),thickness=3)
        else:
            print('Face not found!')
        
        print("cost time = %d"%(time.time() * 1000 - starttime))
        return frame

    def run_camera(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.height)
        while True:
            detected, frame = camera.read()

            if detected:
                frame = cv2.flip(frame, 1)
                frame = self.detective(frame)
            cv2.imshow("AwesomeTang", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def single_image(self,img_path):
        img = self.check_file(img_path)
        img = self.detective(img)
        cv2.namedWindow("AwesomeTang", 0)
        cv2.resizeWindow("AwesomeTang", Config.width, Config.height)
        cv2.imshow("AwesomeTang",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = FaceDetective()
    p.run_camera()