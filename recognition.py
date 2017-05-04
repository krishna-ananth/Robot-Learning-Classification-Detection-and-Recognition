import cv2
import os
import numpy as nmpy

import apc
import homography

trainPath = '/home/krishna/Detection/dataset/'
sift = cv2.SIFT(contrastThreshold=0.04, edgeThreshold=30)

def getAllImages():
    allImages = []
    kpdes = []
    for i in range(12):
        fileList = []
        onekpdesc = []
        for content in os.listdir(trainPath+'%d'%(i+1)):
            img = cv2.imread(trainPath+'%d/'%(i+1)+content, 0)
            fileList.append(img)
            kp1, des1 = sift.detectAndCompute(img, None)
            onekpdesc.append([kp1, des1])
        allImages.append(fileList)
        kpdes.append(onekpdesc)
    return allImages, kpdes

def main():
    
    allImages, allKpDes = getAllImages()
    print 'loaded train images'
    testImages, ids = apc.load_test_data('data/apc_test_3.hdf5')
    print 'loaded test image'

    opfile = open('recognition_result.txt', 'w')
    opfile.write('Id,bb_tl_x,bb_tl_y,bb_lr_x,bb_lr_y\n')

    for (label, objClass) in ids:
        testNum = int(label.split('_')[0])
        objClass = int(objClass)-1
        kp2, des2 = sift.detectAndCompute(testImages[testNum], None)

        found = 0
        x1, y1, x2, y2 = 280, 150, 370, 250
        for j, trainImage in enumerate(allImages[objClass]):
            kp1, des1 = allKpDes[objClass][j]
            numMatches, cood = \
                    homography.matchIt(trainImage, testImages[testNum], \
                    kp1, des1, kp2, des2)
            if cood != None:
                cood = [[cood[0][0], cood[0][1]],[cood[1][0], cood[1][1]]]
                for a in cood:
                    if a[0] < 0: a[0] = 0
                    if a[0] > 640: a[0] = 640
                    if a[1] < 0: a[1] = 0
                    if a[1] > 400: a[1] = 400
                print cood

                if found == 0:
                    ((x1, y1), (x2, y2)) = cood
                else:
                    x1 += cood[0][0]
                    y1 += cood[0][1]
                    x2 += cood[1][0]
                    y2 += cood[1][1]
                found += 1

        if found != 0:
            x1 /= found
            x2 /= found
            y1 /= found
            y2 /= found

        opfile.write('%s,%d,%d,%d,%d\n'%(label, x1, y1, x2, y2))
        print 'done', label

    opfile.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
