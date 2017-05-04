import cv2
import os
import numpy as nmpy
import apc
import homography

trainPath = '/home/krishna/Desktop/Detection/dataset/'
sift = cv2.SIFT(contrastThreshold=0.03, edgeThreshold=10)

def getAllImages():
    allImages = []
    kpdes = []
    for i in range(12):
        fileList = []
        kps = []
        dess = []
        for content in os.listdir(trainPath+'%d'%(i+1)):
            img = cv2.imread(trainPath+'%d/'%(i+1)+content, 0)
            fileList.append(img)
            kp1, des1 = sift.detectAndCompute(img, None)
            print len(kp1), des1.shape
            kps += kp1
            dess.append(des1)
        allImages.append(fileList)
        kpdes.append([kps, nmpy.vstack(dess)])
    return allImages, kpdes

def main():
    
    allImages, allKpDes = getAllImages()
    print 'loaded train images'
    testImages = apc.load_test_data('data/apc_test_3.hdf5')
    print 'loaded test image'

    opfile = open('detection_result.txt', 'w')
    opfile.write('Id,Category\n')

    datFile = open('num_dat.txt', 'w')

    for i, testImage in enumerate(testImages):
        for objClass in range(12):
            kp2, des2 = sift.detectAndCompute(testImage, None)

            found = 0
            #for j, trainImage in enumerate(allImages[objClass]):
            kp1, des1 = allKpDes[objClass]
            trainImage = allImages[objClass]
            numMatches, _ = \
                    homography.matchIt(trainImage, testImage, \
                    kp1, des1, kp2, des2)
            print numMatches, 
            datFile.write('%d\n'%numMatches)
            if numMatches > 10:
                found = 1
                #break

            opfile.write('%d_%d,%d\n'%(i, objClass+1, found))
            print 'done', i, objClass, found

    opfile.close()
    datFile.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
