import numpy as np
import matplotlib.pyplot as plt
import cv2


def correspondingFeatureDetection(img1, img2):

    orb = cv2.ORB_create()
    Keypoints1, Descriptors1 = orb.detectAndCompute(img1, None)
    # Keypoints1, Descriptors1 = orb.compute(img1,Keypoints1)
    # print(Keypoints1)
    Keypoints2, Descriptors2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(Descriptors1,Descriptors2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    kp1_list = []
    kp2_list = []
    for m in matches:

        img1Idx = m.queryIdx
        img2Idx = m.trainIdx

        (img1x, img1y) = Keypoints1[img1Idx].pt
        (img2x, img2y) = Keypoints2[img2Idx].pt

        kp1_list.append((img1x,img1y))
        kp2_list.append((img2x,img2y))
        
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,Keypoints1,img2,Keypoints2,matches[:30],None, flags=2)

    plt.imshow(img3),plt.show()
    return Keypoints1, Descriptors1, Keypoints2, Descriptors2, matches




if __name__ == "__main__":

    img1 = cv2.imread(r'../../mr19-assignment2-data/images/000000.png')
    img2 = cv2.imread(r'../../mr19-assignment2-data/images/000001.png')
    # plt.imshow(img1)
    # plt.show()
    correspondingFeatureDetection(img1, img2)