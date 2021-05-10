import cv2 
import imutils 

class DetecteurMouvement(object):
	def __init__(self, aw=0.5):
		self.aw=aw
		self.bg=None 

	def update(self, img):
		if self.bg is None:
			self.bg=img.copy().astype("float")

		cv2.accumulateWeighted(img, self.bg, self.aw)

	
	def detect(self, img, seuil=30):

		diff=cv2.absdiff(self.bg.astype("uint8"), img)
		tresh=cv2.threshold(diff, seuil, 255, cv2.THRESH_BINARY)[1]
		cnts=cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts=imutils.grab_contours(cnts)
		if len(cnts)==0:
			return None 
		else:
			return (tresh, max(cnts, key=cv2.contourArea))	
