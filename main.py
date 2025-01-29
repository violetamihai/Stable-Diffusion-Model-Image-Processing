
import cv2
import numpy as np
import time
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import moviepy.editor as mpe
from pymediainfo import MediaInfo
import pyttsx3
from pydub import AudioSegment
from moviepy.editor import concatenate_audioclips, AudioFileClip


# cream imaginile care vor porni de la cea originala
imagine = cv2.imread('C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/image.png')
img = cv2.imread('C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/image.png')
img2 = cv2.imread('C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/image.png')
img_cie = cv2.imread('C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/image.png')

# desenare dreptunghi
window_name = 'Imagine cu dreptunghi.'
start_point = (280, 35) 
end_point = (500, 500) 
color = (255, 0, 0) 
thickness = 2
image_rectangle = cv2.rectangle(img2, start_point, end_point, color, thickness) 
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/dreptunghi.png", image_rectangle)

# aplicare masca de culoare
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# definim marginile de culoare in hsv - tot ce nu este negru trece de masca
lower_colors = np.array([0,40,40])
upper_colors = np.array([179,255,255])

mask = cv2.inRange(hsv, lower_colors, upper_colors)
background = cv2.bitwise_and(img,img, mask= mask)

cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/masca.png", mask)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/background.png", background)

# schimbare spatiu de culoare
img_cie2 = background
img_cie = cv2.cvtColor(img_cie, cv2.COLOR_BGR2RGB)
img_cie2 = cv2.cvtColor(img_cie2, cv2.COLOR_BGR2RGB)


#matricea pentru conversia RGB - CIE XYZ
matrix = np.array([[0.412453, 0.357580, 0.180423],
                   [0.212671, 0.715160, 0.072169],
                   [0.019334, 0.119193, 0.95022733]])

# inmultim imaginea in RGB cu matricea
img_cie = np.dot(img_cie, matrix)

img_cie = np.clip(img_cie, 0, 255)
img_cie = img_cie.astype(np.uint8)


img_cie2 = np.dot(img_cie2, matrix)
img_cie2 = np.clip(img_cie2, 0, 255)
img_cie2 = img_cie2.astype(np.uint8)

cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/CIE.png", img_cie)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/CIE2.png", img_cie2)

# adaugare text

imagine1 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/dreptunghi.png")
imagine2 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/background.png")
imagine3 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/masca.png")
imagine4 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/CIE.png")
imagine5 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/image.png")
imagine6 = cv2.imread("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_fara_text/CIE2.png")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5

text_dreptunghi = "Rectangle image"
text_background = "Extracted background"
text_masca = "Mask image"
text_CIE = "CIE XYZ"
text_CIE2 = "CIE XYZ extracted"
text_imagine = "Original image"

cv2.rectangle(imagine1, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine1, text_dreptunghi, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/drepthunghi.png", imagine1)

cv2.rectangle(imagine2, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine2, text_background, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/background.png", imagine2)

cv2.rectangle(imagine3, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine3, text_masca, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/mask.png", imagine3)

cv2.rectangle(imagine4, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine4, text_CIE, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/cie.png", imagine4)

cv2.rectangle(imagine5, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine5, text_imagine, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/imagine.png", imagine5)

cv2.rectangle(imagine6, (150, 400), (350, 450), (255,255,255), thickness=-1)
cv2.putText(imagine6, text_CIE2, (170, 430), font, font_scale, (0,0,0), 2)
cv2.imwrite("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text/cie2.png", imagine6)

# VIDEO
COMPRESSED_FORMAT = True
image_folder = "C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/imagini_cu_text"
video_name = 'C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/video.avi'

if( COMPRESSED_FORMAT == True ):
    # Folosim codec MPEG4 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
else: 
    # Nu folosim comprimare, video-ul va fi salvat frame cu frame.
    fourcc = cv2.VideoWriter_fourcc(*'RGBA')

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, fourcc, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()

# sonor

engine = pyttsx3.init()
speech2="This is the extracted background then it is the new color space then the extracted background with new colorspace, then  the blue rectangle image this is the original image and finally the mask image "
engine.setProperty('rate', 120) 
engine.setProperty('volume', 1)
engine.save_to_file(speech2, "C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/audiouri/audio1.mp3")
engine.runAndWait()


# adaugare sonor
audio = mpe.AudioFileClip("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/audiouri/audio1.mp3")
video1 = mpe.VideoFileClip("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/video.avi")
final = video1.set_audio(audio)
final.write_videofile("C:/Users/Violeta/Desktop/Mihai_Violeta_333AA/video_final.avi",codec='libxvid', audio_codec='libmp3lame')





