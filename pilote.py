import cv2 
import numpy as np 
import imutils 
from detecteur_mouvement import DetecteurMouvement
from descriteur_geste import DetecteurGeste
        

md=DetecteurMouvement()
gd=DetecteurGeste()
stream=cv2.VideoCapture(0)
top, right, bot, left=(10,350,225,590)
nb_image=0
geste=None
valeur=[]
seuil=None
fourcc=cv2.VideoWriter_fourcc(*'MP4V')
writer=cv2.VideoWriter("video4.mp4", 0x7634706d, 13.0, (840, 450))
mask=None
while True:

	continue_, img=stream.read()
	if not continue_:
		break

	img=imutils.resize(img, width=600)
	img=cv2.flip(img, 1)
	copy=img.copy()
	roi=img[top:bot, right:left]	
	gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray=cv2.GaussianBlur(gray, (7, 7), 0)
	if nb_image<32:
		md.update(gray)

	else:
		rs=md.detect(gray)
		if rs is not None:
			(thresh, c) = rs
			seuil=thresh
			cv2.drawContours(copy, [c + (right, top)], -1, (0, 255, 0), 2)
			fingers = gd.detect(thresh, c)

			if geste is None:
				geste = [1, fingers]

			else:
				if geste[1] == fingers:
					geste[0] += 1
 
					if geste[0] >= 25:
						if len(valeur) == 2:
							valeur= []
 
						valeur.append(fingers)
						geste=None
				else:
					geste=None	

	
	if len(valeur)>0:
		DetecteurGeste.drawBox(copy, 0)
		DetecteurGeste.drawText(copy, 0, valeur[0])
		DetecteurGeste.drawText(copy, 1, "+")
 
	if len(valeur)==2:
		DetecteurGeste.drawBox(copy, 2)
		DetecteurGeste.drawText(copy, 2, valeur[1])
		DetecteurGeste.drawText(copy, 3, "=")
		DetecteurGeste.drawBox(copy, 4, color=(0, 255, 0))
		DetecteurGeste.drawText(copy, 4, valeur[0] + valeur[1], color=(0, 255, 0))

	cv2.rectangle(copy, (left, top), (right, bot), (0, 0, 255), 2)
	if seuil is not None:
		if mask is None:
			mask=np.zeros((img.shape[0], img.shape[1]+seuil.shape[1], 3), np.uint8)
		mask[0:img.shape[0], 0:img.shape[1]]=copy
		mask[:, :, 2][0:seuil.shape[0], img.shape[1]:img.shape[1]+seuil.shape[1]]=seuil

	cv2.imshow("stream", copy)
	nb_image+=1
	if mask is not None:	
		writer.write(mask)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break	

stream.release()
writer.release()
cv2.destroyAllWindows()						
