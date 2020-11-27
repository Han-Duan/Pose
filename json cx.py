import numpy as np
import sys
from json import loads
from re import sub
import json
import matplotlib.pyplot as plt
import cv2
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
    "MidHip"
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

body_part_connectors = [
    [17, 15],
    [15, 0],
    [0, 16],
    [16, 18],
    [0, 1],
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [8, 12],
    [9, 10],
    [10, 11],
    [12, 13],
    [13, 14]]

def parseJson():
    with open("./1_keypoints.json", 'r') as f:
        peoples = loads(f.read())['people']  # creates a Python dictionary of Items for the supplied json file
        for people in peoples:
            keyPointArray = people['pose_keypoints_2d']
            keyPoint1 = np.reshape(keyPointArray, [25, 3])
    kp1 = []
    for i in range(0, 24):
        kp1.append([np.int32(keyPoint1[i, 0]), np.int32(keyPoint1[i, 1]), keyPoint1[i, 2], body_part_long_names[i]])

    with open("./2_keypoints.json", 'r') as f:
        peoples = loads(f.read())['people']  # creates a Python dictionary of Items for the supplied json file
        for people in peoples:
            keyPointArray = people['pose_keypoints_2d']
            keyPoint2 = np.reshape(keyPointArray, [25, 3])
    kp2 = []
    for i in range(0, 24):
        kp2.append([np.int32(keyPoint2[i, 0]), np.int32(keyPoint2[i, 1]), keyPoint2[i, 2], body_part_long_names[i]])
    return kp1, kp2


def computeVector(kp1, kp2):
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
        x_diff = kp1[i][0] - kp2[i][0]
        y_diff = kp1[i][1] - kp2[i][1]
        vec_array.append([x_diff, y_diff])
        if int(x_diff) != 0 and int(y_diff) != 0:
            x_sum += x_diff
            y_sum += y_diff
            cnt += 1
    avg_vec = [x_sum / cnt, y_sum / cnt]
    return vec_array, avg_vec

def draw_2d_image_points(
    image_points,
    point_labels=[]):
    image_points = np.asarray(image_points).reshape((-1, 3))
    points_image_u = image_points[:, 0]
    points_image_v = image_points[:, 1]
    points_image_l = image_points[:, 2]
    plt.plot(
        points_image_u,
        points_image_v,
        '.')

    for i in range(len(point_labels)):
        plt.text(points_image_u[i], points_image_v[i], body_part_long_names[points_image_l[i]])

def draw(kp1):
    all_points = []
    for i in range(0,24):
        all_points.append([kp1[i][0], kp1[i][1], i])

    valid_keypoints = np.empty(24,dtype=np.bool_)
    for i in range(0,24):
        if(all_points[i][0] == 0 and all_points[i][1] == 0):
            valid_keypoints[i] = np.False_
        else:
            valid_keypoints[i] = np.True_
    all_points = np.asarray(all_points)
    plottable_points = all_points[valid_keypoints]
    draw_2d_image_points(plottable_points)
    for body_part_connector in body_part_connectors:
        body_part_from_index = body_part_connector[0]
        body_part_to_index = body_part_connector[1]
        if valid_keypoints[body_part_from_index] and valid_keypoints[body_part_to_index]:
            plt.plot(
                [all_points[body_part_from_index][0], all_points[body_part_to_index][0]],
                [all_points[body_part_from_index][1], all_points[body_part_to_index][1]],
                'k-',
                alpha=0.2)
    plt.show()




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
    kp1, kp2 = parseJson()
    print(kp1)
    print(kp2)
    vec, avgVec = computeVector(kp1, kp2)
    print(vec, avgVec)
    imgL = cv2.imread('/Users/handuan/Desktop/untitled folder/WechatIMG5.jpeg',0)
    img2 = cv2.imread('/Users/handuan/Desktop/untitled folder/WechatIMG5.jpeg', cv2.IMREAD_GRAYSCALE)  # queryimage # left image
    plt.imshow(img2)
    draw(kp1)
    kp2_prime = computeKpPrime(kp2, avg_vec)



if __name__ == '__main__':
    main()
