import numpy as nmpy
import cv2

MIN_MATCH_COUNT = 5

#img0 = cv2.imread('20.png', 0)
#img1 = cv2.imread('2.png', 0)
#img2 = cv2.imread('image_57.png', 0)

sift = cv2.SIFT(contrastThreshold=0.01, edgeThreshold=50)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def matchIt(img1, oimg2, kp1, des1, kp2, des2):

    img2 = oimg2.copy()


    #kp0, des0 = sift.detectAndCompute(img0, None)
    #kp1, des1 = sift.detectAndCompute(img1, None)
    #kp2, des2 = sift.detectAndCompute(img2, None)

    #print 'des1:des2', len(des1), ':', len(des2), ' - ',

    #kp1 = kp1 + kp0#nmpy.vstack((kp1, kp0))
    #des1 = nmpy.vstack((des1, des0))

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    return len(good), None

    if len(good)>=MIN_MATCH_COUNT:
        src_pts = nmpy.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = nmpy.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        if mask == None:
            matchesMask = None
        else:
            matchesMask = mask.ravel().tolist()
            #print matchesMask

            h, w = img1.shape
            pts = nmpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            a, b = dst[0][0][0], dst[0][0][1]
            minX, minY, maxX, maxY = a, b, a, b
            for i in range(4):
                x = dst[i][0][0]
                y = dst[i][0][1]
                if x < minX: minX = x
                if x > maxX: maxX = x
                if y < minY: minY = y
                if y > maxY: maxY = y

            '''
            img2 = cv2.polylines(img2,[nmpy.int32(dst)],True,255,3, cv2.LINE_AA)

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

            cv2.imshow('img', img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

            return None, ((minX, minY), (maxX, maxY))

    else:
        #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    
    return len(good), None
