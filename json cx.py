import numpy as np
import sys
from json import loads
from re import sub
import json
import statistics


body_part_long_names = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "Background"]


def parseJson():
    with open("/Users/handuan/WorkSpace/kk/openpose/output_json_folder/1_keypoints.json", 'r') as f:
        peoples = loads(f.read())['people']  # creates a Python dictionary of Items for the supplied json file
        for people in peoples:
            keyPointArray = people['pose_keypoints_2d']
            keyPoint1 = np.reshape(keyPointArray,[25,3])
    kp1 = []
    for i in range(0,24):
        kp1.append([np.int32(keyPoint1[i,0]),np.int32(keyPoint1[i,1]),keyPoint1[i,2], body_part_long_names[i]])

    with open("/Users/handuan/WorkSpace/kk/openpose/output_json_folder/2_keypoints.json", 'r') as f:
        peoples = loads(f.read())['people']  # creates a Python dictionary of Items for the supplied json file
        for people in peoples:
            keyPointArray = people['pose_keypoints_2d']
            keyPoint2 = np.reshape(keyPointArray,[25,3])
    kp2 = []
    for i in range(0,24):
        kp2.append([np.int32(keyPoint2[i,0]),np.int32(keyPoint2[i,1]),keyPoint2[i,2], body_part_long_names[i]])
    return kp1, kp2
    # with open("/Users/handuan/Desktop/COCO_val2014_000000000192_keypoints.json", 'r') as f:
    #     peoples = loads(f.read())['people'] # creates a Python dictionary of Items for the supplied json file
    #     for people in peoples:
    #         keyPointArray = people['pose_keypoints_2d']
    #         for keyPointNumber in range(0, 24):
    #             xIndex = 3 * keyPointNumber
    #             yIndex = xIndex + 1
    #             cIndex = yIndex + 1
    #
    #             xVal = keyPointArray[xIndex]
    #             yVal = keyPointArray[yIndex]
    #             cVal = keyPointArray[cIndex]
    #
    #             nxVal = xVal / np.cos(angle*np.pi/180)
    #             print(str(xVal)+ "->" + str(nxVal))



def computeVector(kp1,kp2):
    if len(kp1) != len(kp2):
        # "length of kp list does not match"
        return -1
    # initialize vector array
    vec_array = []
    x_sum = 0
    y_sum = 0
    cnt = 0
    for i in range(len(kp1)):
        if kp1[i][3] != kp2[i][3]:
            # "key points not match"
            return -2
        x_diff = kp1[i][0]-kp2[i][0]
        y_diff = kp1[i][1]-kp2[i][1]
        vec_array.append([x_diff, y_diff])
        if int(x_diff) != 0 and int(y_diff) != 0:
            x_sum += x_diff
            y_sum += y_diff
            cnt += 1
    avg_vec = [x_sum/cnt, y_sum/cnt]
    return vec_array, avg_vec


# move kp2 to kp1 and call it kp2_prime
def computeKpPrime(kp2, avg_vec):
    kp2_prime = []
    for i in range(len(kp2)):
        if int(kp2[i][0]) != 0 and int(kp2[i][1]) != 0:
            x_tmp = kp2[i][0] - avg_vec[0]
            y_tmp = kp2[i][1] - avg_vec[1]
            confidence = kp2[i][2]
            pos = kp2[i][3]
            kp2_prime.append([x_tmp, y_tmp, confidence, pos])
        else:
            kp2_prime.append(kp2[i])
    return kp2_prime


# method: 
# 1. mean 
# 2. higher confidence
# 3. waited sum
# return list of points (x(int), y(int), position(str))
def selectKp(kp1, kp2_prime, method):
    kp_final = []
    if method == 1:
        for i in range(len(kp1)):
            x_new = round(statistics.mean(kp1[i][0],kp2_prime[i][0]))
            y_new = round(statistics.mean(kp1[i][1],kp2_prime[i][1]))
            kp_final.append([x_new,y_new,kp1[i][3]])
    if method == 2:
        for i in range(len(kp1)):
            # compare the confidence
            if kp1[i][2] >= kp2_prime[i][2]:
                kp_final.append([kp1[i][0], kp1[i][1], kp1[i][3]])
            else:
                kp_final.append([kp2_prime[i][0], kp2_prime[i][1], kp2_prime[i][3]])
    if method == 3:
        pass
    return kp_final
    

def main():
    def main():
    kp1,kp2 = parseJson()
    vec, avg_vec = computeVector(kp1, kp2)
    kp2_prime = computeKpPrime(kp2, avg_vec)


if __name__ == '__main__':
    main()
