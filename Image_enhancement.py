import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img1 = cv2.imread('input1.bmp')
img2 = cv2.imread('input2.bmp')
img3 = cv2.imread('input3.bmp')
img4 = cv2.imread('input4.bmp')
color = ('b','g','r')

def show_histr(img):
    histogram = np.array([])
    #plt.figure(1)
    plt.subplot(2, 1, 1)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        histogram = np.append(histogram,histr)
    #plt.figure(2)
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    histogram = histogram.reshape(3,-1)
    return histogram

def gimp(img, perc = 0.05):
    for channel in range(img.shape[2]):
        mi, ma = (np.percentile(img[:,:,channel], perc), np.percentile(img[:,:,channel],100.0-perc))
        img[:,:,channel] = np.uint8(np.clip((img[:,:,channel]-mi)*255.0/(ma-mi), 0, 255))
    return img
def gray_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)
    
def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y

def HE(img):
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    # Applying equalize Hist operation on Y channel.
    #y_eq = cv2.equalizeHist(y)
    y_eq = histeq(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    #res = np.hstack((img2,img_rgb_eq)) #stacking images side-by-side
    return img_rgb_eq

def unsharp_mask(img,fac):
    img = img.astype(float)
    gmask=img-cv2.GaussianBlur(img,(5,5),0)
    #gmask = gmask.astype(float)
    img[:,:,0] = img[:,:,0] + fac*gmask[:,:,0]
    img[:,:,1] = img[:,:,1] + fac*gmask[:,:,1]
    img[:,:,2] = img[:,:,2] + fac*gmask[:,:,2]
    img = np.where(img>255,255,img)
    img = np.where(img<0,0,img)
    img = img.astype(np.uint8)
    return img
    

#image 1 with power law
out1 = (img1/255.0)**(1.0/2.0)*255.0
cv2.imwrite('output1.bmp',out1)
#out1 = HE(img1)
#cv2.imwrite('output1_HE.bmp',out1)

#image 2 with HE
out2 = HE(img2)
img_HLS = cv2.cvtColor(out2, cv2.COLOR_BGR2HLS)
img_HLS[:,:,2] = (img_HLS[:,:,2]/255.0)**(1.0/1.5)*255.0
out2 = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2BGR)
cv2.imwrite('output2.bmp',out2)
 
#image 3 //discuss noise info tradeoff
out3 = cv2.bilateralFilter(img3,7,55,45)
out3 = (out3/255.0)**(1.0/1.5)*255.0
out3 = out3.astype(np.uint8)
#out3 = HE(img3)
#out3 = cv2.GaussianBlur(out3,(3,3),0)
#out3 = cv2.GaussianBlur(out3,(3,3),0)
out3 = cv2.bilateralFilter(out3,7,55,45)
#out3 = cv2.bilateralFilter(out3,7,55,45)
"""
laplacian = cv2.Laplacian(out3,-1)
out3 = out3.astype(np.int32)
out3[0]=out3[0]-laplacian[0]
out3[1]=out3[1]-laplacian[1]
out3[2]=out3[2]-laplacian[2]
out3 = np.where(out3>255,225,out3)
out3 = np.where(out3<0,0,out3)
out3 = out3.astype(np.uint8)
"""
out3 = unsharp_mask(out3,3)
out3 = (img3/255.0)**(1.0/1.5)*255.0
out3 = out3.astype(np.uint8)
cv2.imwrite('output3.bmp',out3)

#image 4
out4 = gimp(img4,0.05)
#laplacian = cv2.Laplacian(out4,-1)
out4 = out4.astype(np.uint8)
out4 = unsharp_mask(out4,2)
cv2.imwrite('output4.bmp',out4)
