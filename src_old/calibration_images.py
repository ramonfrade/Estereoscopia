
import cv2
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_LEFT = PROJECT_ROOT / "images" / "stereoLeft"
IMAGES_RIGHT = PROJECT_ROOT / "images" / "stereoRight"

IMAGES_LEFT.mkdir(parents=True, exist_ok=True)
IMAGES_RIGHT.mkdir(parents=True, exist_ok=True)

# Silence the Qt logging
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["QT_NO_SESSION_MANAGEMENT"] = "1"

# Silencia os avisos do OpenCV (como garantia)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Inicializa as câmeras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# Força o formato MJPEG
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Define a resolução
largura = 640
altura = 480

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

# Dá 2 segundos para o hardware das Logitech C270 se estabilizar no Linux
print("Aguardando inicialização das câmeras...")
time.sleep(2)

num = 0


while cap1.isOpened():

    success1, img1 = cap1.read()
    success2, img2 = cap2.read()


    k = cv2.waitKey(5)

    if k == 27: # exit if ESC key is pressed
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(str(IMAGES_LEFT / f"imageL{num}.png"), img1)
        cv2.imwrite(str(IMAGES_RIGHT / f"imageR{num}.png"), img2)
        print('images saved!')
        num += 1

    cv2.imshow("Calibracao esq (video0)", img1)
    cv2.imshow("Calibracao dir (video2)", img2)

cap1.release()
cap2.release()
cv2.destroyAllWindows()
