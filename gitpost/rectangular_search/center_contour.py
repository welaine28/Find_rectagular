import argparse
import imutils
import cv2
import os
import numpy as np
import sys
import collections
import math
from scipy.optimize import fsolve
import scipy.optimize as opt
from numpy import exp
import timeit
from matplotlib import pyplot as plt

want='L'

def normalize(f):
	lmin = float(f.min())
	lmax = float(f.max())
	x=(f-lmin)*1.0/(lmax-lmin)*255
	x=x.astype(int)

	return x.astype(np.uint8)

def bitmask(img):
	im= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	im=cv2.bitwise_not(im)
	return im


def adjust_gamma(image, gamma=1.0):
	im = cv2.imread("myblob.jpg", cv2.IMREAD_GRAYSCALE)
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	img=cv2.LUT(image, table)
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgray[np.where((imgray <=20))] = 255
	method_name="adjust_gamma"
	file_name=save_img(filename,method_name,imgray)

	return 
def blob(im):

	#im = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
	im= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	im=cv2.bitwise_not(im)
	method_name="SimpleBlobDetector_create"
	file_name=save_img(filename,method_name,im)

	params = cv2.SimpleBlobDetector_Params()
	#print params
	detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs.
	keypoints = detector.detect(im)
	im=cv2.bitwise_not(im)
	print "keypoints"
	print keypoints

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
 	method_name="blob"
	file_name=save_img(filename,method_name,im_with_keypoints)

def CLAHE_Equalization(img):
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# create a CLAHE object (Arguments are optional).
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(imgray)

	method_name="clahe_2"
	file_name=save_img(filename,method_name,cl1)
	return cl1


def equalization(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')
	plt.show()
  	cv2.equalizeHist(img,dst);

  	#img[np.where(img>20)] 




	def next(cdf):
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')
		img2 = cdf[img]
		method_name="aft_eq"
		file_name=save_img(filename,method_name,img2)


def threshold(img,method_name):
	row,col=img.shape
	imgray = cv2.GaussianBlur(img, (5, 5), 0)
	minVal,maxVal,j,k=cv2.minMaxLoc(imgray)
	threshold=(maxVal+minVal)/2
	ret,thresh = cv2.threshold(imgray,threshold,255,0)
	ret3,th2= cv2.threshold(imgray,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	thresh[0:2,:]=255
	thresh[:,0:2]=255
	thresh[(row-2):row,:]=255
	thresh[:,(col-2):col]=255

	method_name="threshold"+method_name
	file_name=save_img(filename,method_name,thresh)

def denoise(img):
	
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	equ = cv2.equalizeHist(imgray)

	imgray[np.where((imgray <=20))] = 255

	method_name="equ"
	file_name=save_img(filename,method_name,equ)

	method_name="where"
	file_name=save_img(filename,method_name,imgray)
	return imgray



black=False

def save_img(file_name,method_name,img):
	x=filename.find('.')
	y=filename.find('low')
	if y:
		black=True
	path = '/Users/yawang/Documents/gitpost/rectangular_search/output'
	name='_'.join([filename[:x],method_name+'.jpeg'])
	cv2.imwrite(os.path.join(path,name),img)
	return os.path.join(path,name)
def read_img(filename):
	img = cv2.imread(filename)
	row1,col1,num=img.shape
	print "Image Info"
	print "row: %d" %len(img)
	print "col: %d" %len(img[0])
	return img


def dist(p1,p2):
	return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
def mid(p1,p2):
	return (p1[0]+p2[0])/2,(p1[1]+p2[1])/2

def contour(filename,special):
	img = cv2.imread(filename)
	row,col,num=img.shape
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	print "Image Info"
	print "row: %d" %len(img)
	print "col: %d" %len(img[0])
	imgray=normalize(imgray)
	imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
	minVal,maxVal,j,k=cv2.minMaxLoc(imgray)
	threshold=(maxVal+minVal)/2
	ret,thresh = cv2.threshold(imgray,threshold,255,0)

	#kernel = np.ones((5,5), np.uint8)
	#opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)



	method_name="thresh"
	file_name=save_img(filename,method_name,thresh)


	#ret3,th2= cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	

	if special==1:
		th2 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,3)
		th2 = cv2.medianBlur(th2, 3)
		th2 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,4)
		kernel = np.ones((5,5), np.uint8)
		opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
		th2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	elif special ==2:
		th2 = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,3)
		th2 = cv2.medianBlur(th2, 3)
		kernel = np.ones((5,5), np.uint8)
		opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
		th2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	else:
		th2=thresh




	th2[0:2,:]=255
	th2[:,0:2]=255
	th2[(row-2):row,:]=255
	th2[:,(col-2):col]=255

	thresh[0:2,:]=255
	thresh[:,0:2]=255
	thresh[(row-2):row,:]=255
	thresh[:,(col-2):col]=255



	method_name="th2"
	file_name=save_img(filename,method_name,th2)

	im3, contours, hierarchy2 = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if special==1:
		im2, contours2, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	def edge(p,row,col):
		if p[0]>=col-4 or p[0]<=0 or p[1]>=row-4  or p[1]<=0:
			return True
		return False

	def check(p,row,col):
		if p[0]>col:
			p[0]=col-1
		elif p[0]<0:
			p[0]=0
		if p[1]>row:
			p[1]=row-1
		elif p[1]<0:
			p[1]=0
		return p


	area_dic=collections.defaultdict(list)

	for cnt in contours:
		area = cv2.contourArea(cnt)
		area=int(area)
		area_dic[area].append(cnt)

	keylist = area_dic.keys()
	keylist.sort()

	print keylist

	partition=[]
	for i in range(len(keylist)-1,0,-1):
		if keylist[i]<9000:
			break
		if keylist[i]/keylist[i-1]>=2 and keylist[i]/keylist[i-1]<=5:
			partition.append(keylist[i])

	print "partition"
	partition.sort()
	print partition


	partition_dic=collections.defaultdict(list)
	if len(partition)>3:
		partition=partition[:4]

	max_area=1
	mid_contour=contours[0]
	contour_new=[]



	centres = []

	for cnt in contours:
		moments = cv2.moments(cnt)
		if moments['m00']==0:
  			continue
  		center=(int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
		area = cv2.contourArea(cnt)
		area=int(area)
		if area>9000 and area<=max(partition):
			contour_new.append(cnt)
			centres.append(center)
			if area>max_area:
				max_area=area
				mid_contour=cnt

		sorted(centres, key=lambda x: (x[0], x[1]))

	for i in range(len(contour_new)):
  		moments = cv2.moments(contour_new[i])
  		if moments['m00']==0:
  			continue
  		center=(int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
  		


	moments = cv2.moments(mid_contour)
	mid_point=(int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
	cv2.circle(img,(mid_point[0],mid_point[1]),50,(100,220,200),-1)
	
	#center_mid=collections.defaultdict(list)
	ratio=10

	def find(center,centre):
		result=dist(centre[0],centre[-1])
		find=center
		for c in centre:
			if dist(center,c)<result:
				result=dist(center,c)
				find=c
		return result

	if special==1:
		ratio=15
		for cnt in contours2:
			moments = cv2.moments(cnt)
			if moments['m00']==0:
  				continue
  			center=(int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
  			close=find(center,centres)
			area = cv2.contourArea(cnt)
			area=int(area)
			if area>9000 and area<=max(partition) and close*close>area:
				contour_new.append(cnt)


	MAXI_LENGTH=[]


	for cnt in contour_new:

		moments = cv2.moments(cnt)
		if moments['m00']==0:
  			continue
  		center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

  		location =- 1

  		if abs(mid_point[0]-center[0])<math.sqrt(max((partition)))/(ratio) and abs(mid_point[1]-center[1])<math.sqrt(max(partition))/(ratio):
  			location=0
  		elif center[0]<mid_point[0] and center[1]<mid_point[1]:
  			location=2
  		elif center[0]<mid_point[0] and center[1]>=mid_point[1]:
  			location=1
  		elif center[0]>=mid_point[0] and center[1]<mid_point[1]:
  			location=3
  		elif center[0]>=mid_point[0] and center[1]>=mid_point[1]:
  			location=4

		cv2.circle(img,centres[-1],10,255, -1)
		
		length_dic=collections.defaultdict(list)
		point_dic=set()
		x,y,w,h = cv2.boundingRect(cnt)
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		pt_location={}
		count=0
		for point in box:
			point=check(point,row,col)
			x1,y1=point[0],point[1]
			pt_location[(x1,y1)]=count
			cv2.circle(img,(point[0],point[1]),30,(100,220,100),-1)

		maximum_length=-sys.maxint

		for idx in range(len(box)):
			p1=box[idx]
			p2=box[(idx+1)%4]
			p1=check(p1,row,col)
			p2=check(p2,row,col)
			length=dist(p1,p2)
			if length>maximum_length:
				maximum_length=length

		trueArea=int(maximum_length*maximum_length)
		MAXI_LENGTH.append(int(maximum_length*maximum_length))

		size='S'

		if (trueArea<partition[0]) or abs(trueArea-partition[0])<trueArea/20 and special!=1:
			partition_dic[1].append(cnt)

		elif (trueArea<=partition[1] and trueArea>partition[0]) or (abs(trueArea-partition[0])<trueArea/20):
			partition_dic[2].append(cnt)
			size='M'
		elif trueArea>=partition[1]:
			size='L'
			partition_dic[3].append(cnt)



		if want!=size:
			continue



		initial=0
		end=3
		if location==0:
			initial=0
			end=4

		elif location==1:
			initial=2
			end=4

		elif location==2:
			initial=3
			end=5

		elif location==3:
			initial=0
			end=2

		elif location==4:
			initial=1
			end=3
		else:
			initial=0
			end=4

		for idx in range(initial,end):
			p1=box[idx%4]
			p2=box[(idx+1)%4]
			p1=check(p1,row,col)
			p2=check(p2,row,col)
			length=dist(p1,p2)
			x1,y1=p1[0],p1[1]
			x2,y2=p2[0],p2[1]

			if idx%4==2 or idx%4==3:
				length_dic[round(length)].append(((x2,y2),(x1,y1),idx%4))
			else:
				length_dic[round(length)].append(((x1,y1),(x2,y2),idx%4))

		#print "maximum"
		#print maximum_length
			
		for length in length_dic:
			for points in length_dic[length]:
				midx,midy=0,0

				if not(edge(points[0],row,col) or edge(points[1],row,col)) and length>=maximum_length-maximum_length/10:

					midx,midy=mid(points[0],points[1])
					cv2.circle(img,(midx,midy),20,0,-1)
					x1,y1=points[0][0],points[0][1]
					x2,y2=points[1][0],points[1][1]
					point_dic.add((x1,y1))
					point_dic.add((x2,y2))

				else:
					ratio=maximum_length*1.0/dist(points[0],points[1])

					if points[2]==0 or points[2]==2:
						if edge(points[0],row,col) and not edge(points[1],row,col):
							newx=abs(points[1][0]-points[0][0])*ratio+points[1][0]
							newy=abs(points[1][1]-points[0][1])*ratio+points[1][1]
							midx,midy=mid(points[1],(newx,newy))

						elif not edge(points[0],row,col) and edge(points[1],row,col):
							#cv2.circle(img,(points[0][0],points[0][1]),70,(0,80,100),-1)
							#cv2.circle(img,(points[1][0],points[1][1]),70,(0,80,100),-1)
							newx=points[0][0]-abs(points[0][0]-points[1][0])*ratio
							newy=points[0][1]-abs(points[1][1]-points[0][1])*ratio
							midx,midy=mid(points[0],(newx,newy))
					elif points[2] == 1 or points[2] == 3:
						if edge(points[0],row,col) and not edge(points[1],row,col):
							newx=points[1][0]-abs(points[1][0]-points[0][0])*ratio
							newy=abs(points[1][1]-points[0][1])*ratio+points[1][1]
							midx,midy=mid(points[1],(newx,newy))

						elif not edge(points[0],row,col) and edge(points[1],row,col):
							newx=abs(points[1][0]-points[0][0])*ratio+points[0][0]
							newy=abs(points[1][1]-points[0][1])*ratio+points[0][1]
							midx,midy=mid(points[1],(newx,newy))

					if midx>0 and midx<row and midy>0 and midy<col: 
						cv2.circle(img,(int(midx),int(midy)),20,(14,80,155),-1)

	MAXI_LENGTH.sort()
	print MAXI_LENGTH
	#cv2.drawContours(img, contour_new, -1, (0,255,0), 3)


	
  	#print centres
	#cx = int(M['m10']/M['m00'])
	#cy = int(M['m01']/M['m00'])
	method_name="countor_one"
	file_name=save_img(filename,method_name,img)
	







"""
					if points[2]==0 or points[2]==2:
						ratio=maximum_length*1.0/dist(points[0],points[1])
						if edge(points[0],row,col) and not edge(points[1],row,col):
							newx=abs(points[1][0]-points[0]p[0])*ratio+points[1][0]
							newy=abs(points[1][1]-points[0]p[1])*ratio+points[1][1]
							midx,midy=mid(points[1],(newx,newy))

						elif not edge(points[0],row,col) and edge(points[1],row,col):
							newx=points[0][0]-abs(points[0][0]-points[1]p[0])*ratio
							newy=points[0][1]-abs(points[1][1]-points[0]p[1])*ratio
							midx,midy=mid(points[1],(newx,newy))

						
					if points[2]==1 or points[2]==3:
						if edge(points[0],row,col) and not edge(points[1],row,col):
							ratio=maximum_length*1.0/dist(points[0],points[1])

						elif edge(points[0],row,col) and edge(points[1],row,col):
						
"""
						




					



		#for length in length_dic:
		#	if length>maximum_length-3:
		#		for points in length_dic[length]:
		#			if (points[0][0]==points[0][1]):
		#					continue
		#			k=(points[0][1]-points[1][1])/(points[0][0]-points[0][1])
		#			b=points[0][1]-points[0][0]*k
		#			for point in points:
		#				if point in point_dic:
		#					location=pt_location[point]












			

		#cv2.drawContours(img,[box],0,(120,0,255),10)
		#cv2.rectangle(img,(x,y),(x+w,y+h),(120,120,0),2)

	





filename = 'pic1.jpeg'



#threshold(img,"gamma")

#cl1=CLAHE_Equalization(imgray)
#bit=bitmask(img)
#threshold(bit,"cl1")

#imgray=denoise(img)


#equalization(imgray)
contour(filename,0)
#blob(img)
#contour_center(filename)
