import numpy as np
import cv2
cap = cv2.VideoCapture('solidWhiteRight.mp4')
def convert_to_HSV(frame):
 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 return hsv
def display_lines(image, lines):
 line_image = np.zeros_like(image)
 if lines is not None:
 for line in lines:
 x1, y1, x2, y2 = line.reshape(4)
 cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
 return line_image
def perspectiveWarp(inpImage):
 spt_A = [205, 0]
 spt_B = [439, 0]
 spt_C = [0, 205]
 spt_D = [596, 205]
 src = np.float32([spt_A, spt_B, spt_C, spt_D])
 dst = np.float32([(160, 0), (480, 0), (160, 640), (480, 640)])
 matrix = cv2.getPerspectiveTransform(src, dst)
 minv = cv2.getPerspectiveTransform(dst, src)
 birdseye = cv2.warpPerspective(inpImage, matrix, (640, 640))
 return birdseye, minv
23
def trajectoryCalc(list, center, heightdif, height):
 targetPoints = []
 for index in range(len(list)-2): # 0..2
 targetPoints.append(list[index+1])
 # print(list[index+1])
 # print(targetPoints)
 targetX = int(sum(targetPoints) / len(targetPoints))
 #targetY = int(height/2)
 targetVector = [(center, height), (targetX, 0)]
 return targetVector, targetX # [(x,y) , (x,y)]
def trajectoryimg(img, target):
 offset = 320
 birdeyeOffset = 20
 cv2.line(img, (target[0][0]+birdeyeOffset, target[0][1]+offset),
 (target[1][0], target[1][1]+offset), (255, 255, 255), 4)
 return img
def getLaneinfo(img):
 testHeight, testWidth = img.shape[:2]
 print("Height: ", testHeight, "Width: ", testWidth) # 80 x 640
 Left = []
 Right = []
 for row in range(0, testHeight, 10):
 leftLanefound = False
 rightLanefound = False
 col = 321
 while not leftLanefound and col > 0:
 if img[row, col] > 180:
 Left.append((row, col))

 leftLanefound = True
 break
 col = col - 1
 col = 321
 while not rightLanefound and col < 640:
 if img[row, col] > 180:
 Right.append((row, col))
 rightLanefound = True
 break
 col = col + 1
 if not leftLanefound and rightLanefound:
 avgRightX, avgRightY = int(sum(i[1] for i in Right) / len(Right)), int(sum(i[0]
 for i in Right) / len(Right))
 avgLeftY = avgRightY
 if avgRightX > 260:
 avgLeftX = avgRightX - 260
 else:
 avgLeftX = 0
 elif leftLanefound and not rightLanefound:
 avgLeftX, avgLeftY = int(sum(i[1] for i in Left) / len(Left)), int(sum(i[0]
 for i in Left) / len(Left))
 avgRightY = avgLeftY
 if avgLeftX < 360:
 avgRightX = avgLeftX + 260
 else:
 avgRightX = 640
 elif not leftLanefound and not rightLanefound:
 avgLeftX, avgLeftY = 190, 45
 avgRightX, avgRightY = 445, 40
 print("no lane")
 else:
 avgLeftX, avgLeftY = int(sum(i[1] for i in Left) /
 len(Left)), int(sum(i[0] for i in Left) / len(Left))
 avgRightX, avgRightY = int(sum(i[1] for i in Right) / len(Right)),
int(sum(i[0]for i in Right) / len(Right))
 point = (int((avgLeftX+avgRightX) / 2), int((avgLeftY + avgRightY) / 2))

 #cv2.circle(img, (point[0],point[1]), 10, (255, 255, 255), cv2.FILLED)
 print("avgLeftX: ", avgLeftX, "avgLeftY: ", avgLeftY, "avgRightX: ", avgRightX,
"avgRightY: ", avgRightY)
 #print("point: ",point)
 return point[0]
def regionCrop(img, count):

 # Init List of MIddlepoints, Crop birdview image to half of height
 middlePoints = []
 regionHeight, regionWidth = img.shape[:2] # 640 x 640
 print("regionHeight: ",regionHeight, "regionWidth: ", regionWidth)
 crop_img = img[int(regionHeight/2): regionHeight, 0: regionWidth]
 cropHeight, cropWidth = crop_img.shape[:2] # 320 x 640
 print("cropHeight: ",cropHeight, "cropWidth: ", cropWidth)
 countHeight = round(cropHeight / count)
 print("countHeight: ", countHeight)
 for index in range(count):
 crop_bird = crop_img[int(cropHeight-(index+1)*countHeight): int(cropHeightindex*countHeight), 0: cropWidth]
 middlePoint = getLaneinfo(crop_bird)
 middlePoints.append(middlePoint)
 return middlePoints, countHeight, cropHeight
def detect_lanes_img(img):

 # Init Paramters
 height, width = img.shape[:2] # 480 x 640

 # Get ROI + Cutting Borders caused by undistort
 roi_img = img[int(height * 0.4):height-5, 0: width-10]
 roi_height, roi_width = roi_img.shape[:2] # 283 x 630
 hsv = convert_to_HSV(img)

 # Get Lane Information
 # Convert to grayimage
 gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

 # Apply gaussian filter
 blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

 # Apply Canny edge transform
 canny_img = cv2.Canny(blur_img, 100, 130)

 # Apply Threshold Function
 thresh_img = cv2.threshold(gray_img, 64, 255, cv2.THRESH_BINARY_INV)

 # Combine Canny and Threshold for better result
 combo_image = cv2.addWeighted(thresh_img, 1, canny_img, 1, 1)

 # Apply Hough Transformation
 lines = cv2.HoughLinesP(canny_img, 2, np.pi/180, 10, np.array([]),
minLineLength=100, maxLineGap=60)
 line_image = display_lines(roi_img, lines)
 lane_image = cv2.addWeighted(roi_img, 1, line_image, 1, 1)
 birdView, minverse = perspectiveWarp(combo_image)

 realCenter = 302 # wahre Mitte des Fahrzeugs
 cv2.line(lane_image, (realCenter, roi_height),
 (realCenter, roi_height-80), (225, 255, 0), 2)

 nIntervalls = 4
 middlePoints, IntervallHeight, cropHeight = regionCrop(
 birdView, nIntervalls)
 print("middlePoints: ", middlePoints, "IntervallHeight: ", IntervallHeight)
 target, targetX = trajectoryCalc(
 middlePoints, realCenter, IntervallHeight, cropHeight)

 birdView2, minverse = perspectiveWarp(roi_img)
 traject_bird = trajectoryimg(birdView2, target)
 traject_image = cv2.warpPerspective(traject_bird, minverse, (630, 200))

 cv2.imshow("guassian", blur_img)
 cv2.imshow("image_undistort", image_undistort)
 cv2.imshow("hsv",hsv)
 cv2.imshow("traject_bird", traject_bird)
 cv2.imshow("traject_image", traject_image)
 cv2.imshow("birdView2", birdView2)
 cv2.imshow("birdView", birdView)
 cv2.imshow("lane_image", lane_image) #****** hough transform **
 cv2.imshow("roi_img", roi_img)
 cv2.imshow("cannyroi", canny_img)
 #cv2.imshow("threshroi", thresh_img)
 cv2.imshow("combo_image", combo_image)
 return realCenter, targetX
if __name__ == '__main__':
 while (cap.isOpened()):

 ret, frame = cap.read()

 result = detect_lanes_img(frame)
 key = cv2.waitKey(1)
 if key == 27:
 break
 cap.release()
 cv2.destroyAllWindows()
