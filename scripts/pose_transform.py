#import cv2
import numpy as np
#import torch
#import sys

class translate(object):
    """
    S is the joint-list of source object.
    R is the dictionary of ratio: target/source
    origin : [Source original point, target original point]
    ratio_body: target height / source height
    :return: run self.run() target coordinate
    """

    def __init__(self, S, R, origin, ratio_body):
        # trasnform list to dictionary
        self.S_dic = {}
        self.S_coordinate = {}
        self.R = R
        self.T_coordinate = {}
        self.ratio_body = ratio_body
        self.S_origin, self.T_origin = origin[0], origin[1]

        assert len(S) == 18  # make sure we have 18 arrays, even we may have empty array
        for i in range(18):
            self.S_dic[i] = []
            self.S_coordinate[i] = []
            self.T_coordinate[i] = []

        """
        self.S_max_height = 0
        self.S_min_height = S[0][0][1]  # get the first y from s
        self.T_max_height = 0
        self.T_min_height = S[0][0][1]
        """
        for si in range(18):
            if len(S[si]) == 0:
                self.S_dic[si] = None
                self.S_coordinate[si] = None
                continue

            for i in range(len(S[si])):  # !! in case n has more than 1 element !!

                n = S[si][i]
                self.S_dic[si].append(n)
                self.S_coordinate[si].append(np.array([n[0], n[1]]))
                #self.S_max_height = max(self.S_max_height, n[1])
                #self.S_min_height = min(self.S_min_height, n[1])

    def calc_coordinate(self, start, end, ratio):
        if not self.S_coordinate[start] or not self.S_coordinate[end]:
            self.T_coordinate[end] = self.S_coordinate[end]
            # raise ValueError('source coordinate is missing, index is {} and {}'.format(start, end))

        else:
            if len(self.S_coordinate[start]) >= len(self.S_coordinate[end]):
                for i in range(len(self.S_coordinate[end])):
                    self.T_coordinate[end].append(
                        self.R[ratio] * (self.S_coordinate[end][i] - self.S_coordinate[start][i]) + \
                        self.T_coordinate[start][i])
                    #self.T_max_height = max(self.T_max_height, self.T_coordinate[end][i][1])
                    #self.T_min_height = min(self.T_min_height, self.T_coordinate[end][i][1])
            else:
                for i in range(len(self.S_coordinate[end])):
                    if i < len(self.S_coordinate[start]):
                        self.T_coordinate[end].append(
                            self.R[ratio] * (self.S_coordinate[end][i] - self.S_coordinate[start][i]) + \
                            self.T_coordinate[start][i])
                    else:
                        self.T_coordinate[end].append(
                            self.R[ratio] * (self.S_coordinate[end][i] - self.S_coordinate[start][-1]) + \
                            self.T_coordinate[start][-1])
                    #self.T_max_height = max(self.T_max_height, self.T_coordinate[end][i][1])
                    #self.T_min_height = min(self.T_min_height, self.T_coordinate[end][i][1])

    def local_proportion(self):
        # fix nose
        if not self.S_coordinate[1]:
            raise ValueError('Neck pose (no.1) is missing !')
        S_origin_list = np.array([self.S_origin for i in range(len(self.S_coordinate[1]))])
        T_origin_list = np.array([self.T_origin for i in range(len(self.S_coordinate[1]))])
        self.T_coordinate[1] = list(self.ratio_body*(np.array(self.S_coordinate[1]) - S_origin_list) + \
                                    T_origin_list)

        # neck
        self.calc_coordinate(1, 0, '0-1')
        # right
        self.calc_coordinate(0, 14, '0-14')  # eye
        self.calc_coordinate(14, 16, '14-16')  # ear
        self.calc_coordinate(1, 2, '1-2')  # shoulder
        self.calc_coordinate(2, 3, '2-3')  # elbow
        self.calc_coordinate(3, 4, '3-4')  # wrist
        self.calc_coordinate(1, 8, '1-8')  # hip
        self.calc_coordinate(8, 9, '8-9')  # knee
        self.calc_coordinate(9, 10, '9-10')  # ankle
        # left
        self.calc_coordinate(0, 15, '0-14')  # eye
        self.calc_coordinate(15, 17, '14-16')  # ear
        self.calc_coordinate(1, 5, '1-2')  # shoulder
        self.calc_coordinate(5, 6, '2-3')  # elbow
        self.calc_coordinate(6, 7, '3-4')  # wrist
        self.calc_coordinate(1, 11, '1-8')  # hip
        self.calc_coordinate(11, 12, '8-9')  # knee
        self.calc_coordinate(12, 13, '9-10')  # ankle

    def globalscale(self):
        # compute scale
        all_R = (self.S_max_height - self.S_min_height) / (self.T_max_height - self.T_min_height)
        # scaling fixing nose
        for i in range(1, 18):
            if self.T_coordinate[i]:
                for j in range(len(self.T_coordinate[i])):
                    # if we have multiple T_coordinate[0], we choose the first one.
                    self.T_coordinate[i][j] = all_R * (self.T_coordinate[i][j] - self.T_coordinate[0][0]) \
                                              + self.T_coordinate[0][0]

    def run(self):
        self.local_proportion()
        """
        T1 = []
        for i in self.S_dic.keys():
            if i in range(0, 18):
                T1.append(list(self.T_coordinate[i]) + list(self.S_dic[i][2:]))
            else:
                T1.append(self.S_dic[i])
        """
        #self.globalscale()
        T2 = []
        for i in range(0, 18):

            if self.T_coordinate[i]:
                tmp = []
                for j in range(len(self.T_coordinate[i])):
                    new = list(self.T_coordinate[i][j]) + list(self.S_dic[i][j][2:])
                    tmp.append(new)
                T2.append(np.array(tmp))
            else:
                T2.append(np.array([]))

        # return np.array(T1),np.array(T2)
        return T2

    
if __name__ == '__main__':
    src = '../data/source/ratio_a.png'
    tar = '../data/target/ratio_b.png'
    
    #ratio = pose_ratio(src, tar)
    
    ratio = {'0-1': 2.3, '1-2': 1.7, '2-3': 1.35, '3-4': 1.5, '1-8': 1.61, '8-9': 1.21, '9-10': 0.88, '0-14':1, '14-16':1}
    #S = [np.array([[263.        , 130.        ,   0.93236803,   0.        ]]), np.array([[257.        , 170.        ,   0.86081196,   1.        ]]), np.array([[227.        , 174.        ,   0.75210118,   2.        ]]), np.array([[188.        , 215.        ,   0.79342183,   3.        ]]), np.array([[170.        , 174.        ,   0.78027675,   4.        ]]), np.array([[287.        , 167.        ,   0.79779556,   5.        ]]), np.array([[296.        , 219.        ,   0.84240457,   6.        ]]), np.array([[265.        , 204.        ,   0.66163531,   7.        ]]), np.array([[229.        , 264.        ,   0.44615341,   8.        ]]), np.array([[233.        , 367.        ,   0.66888687,   9.        ]]), np.array([[220.        , 461.        ,   0.55017908,  10.        ]]), np.array([[267.        , 266.        ,   0.45116855,  11.        ]]), np.array([[266.        , 366.        ,   0.76933903,  12.        ]]), np.array([[247.        , 420.        ,   0.44936439,  13.        ]]), np.array([[254.        , 123.        ,   0.93804097,  14.        ]]), np.array([[268.        , 123.        ,   0.92165394,  15.        ]]), np.array([[238.       , 129.       ,   0.9245356,  16.       ]]), np.array([])]
    S = [np.array([[2.1100000e+02, 7.8000000e+01, 8.8372922e-02, 0.0000000e+00]]), np.array([[121.        , 258.        ,   0.80313456,   1.        ]]), np.array([[141.        , 256.        ,   0.34916986,   2.        ]]), np.array([[154.        , 288.        ,   0.31572672,   3.        ]]), np.array([[155.        , 320.        ,   0.34377026,   4.        ]]), np.array([[103.        , 262.        ,   0.34413617,   5.        ]]), np.array([[1.03000000e+02, 2.87000000e+02, 2.62220328e-01, 6.00000000e+00]]), np.array([[1.55000000e+02, 3.20000000e+02, 2.78355942e-01, 7.00000000e+00]]), np.array([[1.39000000e+02, 3.21000000e+02, 2.94884249e-01, 8.00000000e+00]]), np.array([[1.51000000e+02, 3.72000000e+02, 3.58562458e-01, 9.00000000e+00]]), np.array([[1.4200000e+02, 4.2600000e+02, 3.6958338e-01, 1.0000000e+01]]), np.array([[1.37000000e+02, 3.16000000e+02, 1.73621560e-01, 1.10000000e+01],
       [1.12000000e+02, 3.21000000e+02, 2.77241905e-01, 1.20000000e+01]]), np.array([[1.11000000e+02, 3.73000000e+02, 3.17529887e-01, 1.30000000e+01]]), np.array([[1.16000000e+02, 4.31000000e+02, 3.61633401e-01, 1.40000000e+01]]), np.array([[2.0900000e+02, 7.2000000e+01, 1.1098569e-01, 1.5000000e+01]]), np.array([[2.1600000e+02, 7.4000000e+01, 1.0123766e-01, 1.6000000e+01]]), np.array([[1.2600000e+02, 2.3900000e+02, 3.1396721e-01, 1.7000000e+01],
       [1.0900000e+02, 2.4400000e+02, 2.2428293e-01, 1.8000000e+01]]), np.array([[108.        , 242.        ,   0.3559613 ,  19.        ],
       [124.        , 242.        ,   0.28289524,  20.        ]])]

    new=translate(S,ratio)
    new1=new.run()
    print(new1)

    
