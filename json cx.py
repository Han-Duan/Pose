import numpy as np
import sys
from json import loads
from re import sub
import json



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






def main():
    k,t = parseJson()
    a= 1


if __name__ == '__main__':
    main()
