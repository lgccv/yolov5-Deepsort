from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from filterpy.kalman import KalmanFilter


# SORT 多目标跟踪，管理了多个卡尔曼滤波器
class Sort(object):
    # 设置SORT算法的参数
    def __init__(self,max_age=1,min_hits=3):
        # 最大的未检出帧数
        self.max_age = max_age
        # 最小命中次数
        self.min_hits = min_hits
        # 跟踪器
        self.trackers = []
        # 帧计数
        self.frame_count = 0

    def update(self,dets):
        self.frame_count +=1
        # 存储跟踪器对于当前帧图像的预测
        trks = np.zeros((len(self.trackers),5))
        # 要删除的目标框
        to_del = []
        # 返回的跟中目标
        ret = []
        # 遍历卡尔曼滤波器中的跟踪框
        for t ,trk in enumerate(trks):
            # 使用卡尔曼滤波器对目标进行预测
            pos = self.trackers[t].predict()[0]
            # 将预测结果更新到trt中
            trk[:]=[pos[0],pos[1],pos[2],pos[3],0]
            # 若预测pos中包含空，添加到del中
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # trks中去除无效值的行，保存了根据上一帧结果预测的当前帧的内容
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # 删除nan
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 使用匈牙利算法:将检测框和跟踪框进行匹配，跟踪成功的目标，新增的目标，离开的目标
        matched,unmatched_dets,unmatch_trks = associate_detection_to_tracker(dets,trks)
        # 跟踪成功更新到对应的卡尔曼滤波
        for t,trk in enumerate(self.trackers):
            if t not in unmatch_trks:
                d = matched[np.where(matched[:,1] ==t)[0],0]
                # 使用检测框更新卡尔曼滤波器的状态变量
                trk.update(dets[d:][0])
        # 新增目标创建新的卡尔曼滤波器
        for i in unmatched_dets:
            trk = KalmanFilter()
            self.trackers.append(trk)
        i = len(self.trackers)

        # 逆向遍历
        for trk in reversed(self.trackers):
            # 当前边界框的估计值
            d =trk.get_state()[0]
            # 跟踪成功目标box,id进行返回


# 将yolo模型的检测框和卡尔曼滤波的跟踪框进行匹配
def associate_detection_to_tracker(detections,trackers,iou_threshold=0.3):
    '''
    跟踪成功的目标
    新增目标
    跟踪失败的目标
    '''
    if len(trackers) == 0 or (len(detections)==0):
        return np.empty((0,2),dtype=int),np.arange(len(detections)),np.empty((0,5),dtype=int)
    #IOU 逐个IOU计算，构造矩阵,scipy_linear_assignment进行匹配
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float)
    # 遍历检测框
    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    # 调用linear_assigments进行匹配
    result = linear_sum_assigment(-iou_matrix)





