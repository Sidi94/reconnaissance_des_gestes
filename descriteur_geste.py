import cv2 
import numpy as np 
import imutils 
from sklearn.metrics import pairwise 

class DetecteurGeste(object):
	def __init__(self):
		pass 


	def detect(self, thresh, cnt):

		hull = cv2.convexHull(cnt)
		extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
		extRight = tuple(hull[hull[:, :, 0].argmax()][0])
		extTop = tuple(hull[hull[:, :, 1].argmin()][0])
		extBot = tuple(hull[hull[:, :, 1].argmax()][0])	
		cX = (extLeft[0] + extRight[0]) // 2
		cY = (extTop[1] + extBot[1]) // 2
		cY += (cY * 0.15)
		cY = int(cY)

		D = pairwise.euclidean_distances([(cX, cY)], Y=[extLeft, extRight, extTop, extBot])[0]
		maxDist = D[D.argmax()]
		r = int(0.7 * maxDist)
		circum = 2 * np.pi * r
 
		circleROI = np.zeros(thresh.shape[:2], dtype="uint8")
		cv2.circle(circleROI, (cX, cY), r, 255, 1)
		circleROI = cv2.bitwise_and(thresh, thresh, mask=circleROI)
		#cv2.imshow("circle", circleROI)
		cnts = cv2.findContours(circleROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)
		total = 0
 
		for c in cnts:
			(x, y, w, h) = cv2.boundingRect(c)
			if c.shape[0] < circum * 0.25 and (y + h) < cY + (cY * 0.25):
				total += 1
 
		return total

	
	@staticmethod
	def drawText(roi, i, val, color=(0, 0, 255)):
		cv2.putText(roi, str(val), ((i * 50) + 20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			color, 3)

	@staticmethod
	def drawBox(roi, i, color=(0, 0, 255)):
		cv2.rectangle(roi, ((i * 50) + 10, 10), ((i * 50) + 50, 60), color, 2)

