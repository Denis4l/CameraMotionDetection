import cv2
import numpy as np
import time
import datetime
from threading import Thread

"""
Documentation sur OpenCV : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
MotionDetection Class, permet de détecter des mouvements via une camera :
    path: là ou vous voulez enregistrer vos fichiers
    video_source: valeur numérique d'identiication de la camera (commence avec un 0)
    video_size: chaîne de caractères définissant la résolution (ie '1080p')
    threshold: seuil -> 0 très sensible ; 100000 insensible
    time_interval: intervalles d'enregistrement
    recording_time: temps d'enregistrement de la vidéo
    show_camera: booléen indicant la capture
    show_mask: booléen indiquant un mouvement
    debug: booléen pour l'indication du niveau de bruit
"""
class MotionDetection(object):
    is_recording = False
    time_counter = 0

    # Dimensions standard des Video
    STD_DIMENSIONS = {
        "480p": (640, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }

    def __init__(self, path, video_source, video_size, threshold, time_interval, recording_time, show_camera, show_mask, debug):
        self.video_source = video_source                # Source de la video
        self.video_size = video_size                    # taille video
        self.get_dimensions(video_size)                 # largeur et hauteur de l'image
        self.threshold = threshold                      # Niveau de bruit maximum
        self.time_interval = time_interval              # Intervalle entre enregistrements in seconds
        self.recording_time = recording_time            # Durée d'enregistrement en cas de détection de mouvement
        self.path = path                                # lieu de suavegarde de la video
        self.show_camera = show_camera
        self.show_mask = show_mask
        self.debug = debug

        # Initialisations
        self.cap = cv2.VideoCapture(video_source)
        self.sub = cv2.createBackgroundSubtractorMOG2()
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

        # Initialisation de la fenêtre
        if self.show_camera:
            cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Motion Detection', 640, 420)

        if self.show_mask:
            cv2.namedWindow('Motion Mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Motion Mask', 640, 420)

    def get_dimensions(self, video_size):
        self.width, self.height = self.STD_DIMENSIONS['480p']
        if video_size in self.STD_DIMENSIONS:
            self.width, self.height = self.STD_DIMENSIONS[video_size]

    def start(self):
        self.time_counter = time.time()
        print("Camera detected! Size: " + str(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 'x' + str(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("Motion Detection activated. Waiting for motion...")
        while (True):
            # Lecture
            ret, frame = self.cap.read()
            if self.show_camera:
                cv2.imshow("Motion Detection", frame)

            # Création d'un masque
            blur = cv2.GaussianBlur(frame, (19, 19), 0)
            mask = self.sub.apply(blur)
            if self.show_mask:
                cv2.imshow("Motion Mask",mask)

            # Création de l'histogramme de numpy pour annalyse du bruit
            img_temp = np.ones(frame.shape, dtype="uint8") * 255
            img_temp_and = cv2.bitwise_and(img_temp, img_temp, mask=mask)
            img_temp_and_bgr = cv2.cvtColor(img_temp_and, cv2.COLOR_BGR2GRAY)
            hist, bins = np.histogram(img_temp_and_bgr.ravel(), 256, [0, 256])
            if self.debug:
                print("Threshold =", self.threshold, ", Noise = ", hist[255], )

            # Comparaison de l'histogramme et du niveau de bruit
            # Lancement de l'enregistrement
            if hist[255] > self.threshold and not self.is_recording and time.time() - self.time_counter > self.time_interval:
                print("Motion detected! Recording video...")
                self.is_recording = True
                record_thread = Thread(target=self.record_video)
                record_thread.start()
            if cv2.waitKey(100) == 13:
                break

    # Enregistrement
    def record_video(self):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(self.path + '\\' + date+'.mp4', fourcc, 29.0, self.STD_DIMENSIONS[self.video_size])
        time_counter = time.time()
        while time.time() - time_counter < self.recording_time and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == True:
                out.write(frame)
            else:
                break
        print("Video recorded: " + self.path + '\\' + date + '.mp4')
        self.time_counter = time.time()
        self.is_recording = False

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


MD = MotionDetection("D:\\Projets\\borneGonflage\\local", 1, '1080p', 15000, 5, 1, True, True, False)
MD.start()
MD.end()
