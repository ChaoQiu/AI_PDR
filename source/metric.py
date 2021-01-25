import math

import numpy as np


def compute_absolute_trajectory_error(est, gt):
    return np.sqrt(np.mean((est - gt) ** 2))


def Nrotate(angle, valuex, valuey):
    nRotatex = (valuex - valuex[0]) * math.cos(angle) - (valuey - valuey[0]) * math.sin(angle) + valuex[0]
    nRotatey = (valuex - valuex[0]) * math.sin(angle) + (valuey - valuey[0]) * math.cos(angle) + valuey[0]
    return np.concatenate([nRotatex,nRotatey],axis=1)

def compute_relative_50_error(est, gt, delta,duiqi, max_delta=-1):
    deltas_est=[]
    if max_delta == -1:
        max_delta = est.shape[0]
    # deltas = np.array(delta) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    deltas = np.array(delta)
    # for i in range(deltas.shape[0]):
    #     left=right=-1
    #     for j in range(min(duiqi.shape[0],deltas.shape[0])):
    #         if deltas[i,0]<duiqi[j,0] and left==-1:
    #             if j==0:
    #                 left=deltas[i,0]
    #             else:
    #                 left=deltas[i,0]+duiqi[j-1,1]-duiqi[j-1,0]
    #         if deltas[i,1]<duiqi[j,0] and right==-1:
    #             if j==0:
    #                 right=deltas[i,1]
    #             else:
    #                 right=deltas[i,1]+duiqi[j-1,1]-duiqi[j-1,0]
    #         if j==duiqi.shape[0]-1 or (left==-1 and right==-1):
    #             if left==-1:
    #                 left=deltas[i,0]
    #             if right==1:
    #                 right=deltas[i,1]
    #             deltas_est.append([left,right])
    #             break
    # deltas_est=np.array(deltas_est)

    deltas_est=deltas
    # rtes = np.zeros(deltas.shape[0])
    # for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
    # err = (est[deltas[:,1]]-est[deltas[:,0]])-(gt[deltas[:,1]]-gt[deltas[:,0]])

    # est_distance = np.sqrt(np.sum((est[deltas_est[:,1]]-est[deltas_est[:,0]])**2,axis=1))
    # gt_distance = np.sqrt(np.sum((gt[deltas[:,1]]-gt[deltas[:,0]])**2,axis=1))
    # rte = np.mean(abs(est_distance-gt_distance))

    # 取前后10帧
    est_distances = []
    rtes = []
    CEP_68s=[]
    for i in range(-10, 10):
        est_distances.append(np.sqrt(np.sum((est[deltas[:, 1] + i] - est[deltas[:, 0]]) ** 2, axis=1)))
        CEP_68s.append(np.mean(abs(np.sqrt(np.sum((est[deltas_est[:,1]+i]-gt[deltas[:,1]])**2,axis=1)))))
    gt_distance = np.sqrt(np.sum((gt[deltas[:, 1]] - gt[deltas[:, 0]]) ** 2, axis=1))
    for i in range(len(est_distances)):
        rtes.append(np.mean(abs(est_distances[i] - gt_distance)))
    rte = min(rtes)
    CEP_68 = min(CEP_68s)

    # The average of RTE of all window sized is returned.
    return rte,CEP_68
def compute_relative_trajectory_error(est, gt, delta, max_delta=-1):

    if max_delta == -1:
        max_delta = est.shape[0]
    deltas = np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    rtes = np.zeros(deltas.shape[0])
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[deltas[i]:] + gt[:-deltas[i]] - est[:-deltas[i]] - gt[deltas[i]:]
        rtes[i] = np.sqrt(np.mean(err ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(rtes)

def compute_ate_rte(est, gt, pred_per_min=1200):
    ate = compute_absolute_trajectory_error(est, gt)
    if est.shape[0] < pred_per_min:
        ratio = pred_per_min / est.shape[0]
        rte = compute_relative_trajectory_error(est, gt, delta=est.shape[0] - 1) * ratio
    else:
        rte = compute_relative_trajectory_error(est, gt, delta=pred_per_min)

    return ate, rte


def compute_heading_error(est, gt):
    mse_error = np.mean((est - gt) ** 2)
    dot_prod = np.sum(est * gt, axis=1)
    angle = np.arccos(np.clip(dot_prod, a_min=-1, a_max=1))

    return mse_error, angle

def cor_rotation(est,gt):
    metri_gt = 0
    for i in range(len(gt)-1):
        metri_gt += np.sqrt((gt[i, 0] - gt[i + 1, 0]) ** 2 + (gt[i, 1] - gt[i + 1, 1]) ** 2)
        if metri_gt>5:
            init_angel_error = get_heading(gt[0], gt[i]) - get_heading(est[0], est[i])
            if init_angel_error > 180:
                init_angel_error = init_angel_error - 360
            if init_angel_error < -180:
                init_angel_error = init_angel_error + 360
            # init_angel_error=0
            gt = Nrotate(math.radians(-init_angel_error), gt[:, 0:1], gt[:, 1:2])
            return gt,init_angel_error


def compute_step_error(est, gt):
    error_angel = []
    error_angel_no_wan = []
    error_distance = []
    CEP_distance = []
    gt_headings = []
    est_headings = []
    gt_headings_var = []
    est_headings_var = []
    sum_dis_error = []
    ind_15 = []
    ind_wan = []
    ind_wan_est = []
    ind = []
    duiqi=[]
    metri_gt = 0
    metri_est = 0
    temp = 0
    gt_sum_dis = 0
    est_sum_dis = 0
    sum_dis = []
    label_50=-1
    dis_error_50_offset=100
    for i in range(len(gt)-1):
        dgt=np.sqrt((gt[i, 0] - gt[i + 1, 0]) ** 2 + (gt[i, 1] - gt[i + 1, 1]) ** 2)
        dest=np.sqrt((est[i, 0] - est[i + 1, 0]) ** 2 + (est[i, 1] - est[i + 1, 1]) ** 2)
        metri_gt += dgt
        metri_est += dest
        gt_sum_dis += dgt
        est_sum_dis +=dest
        sum_dis.append([gt_sum_dis,est_sum_dis])
        if metri_gt > 0.75:
            sum_dis_error.append([i, gt_sum_dis - est_sum_dis])
            gt_heading = get_heading(gt[temp], gt[i])
            gt_headings.append(gt_heading)
            pre_heading = get_heading(est[temp], est[i])
            est_headings.append(pre_heading)
            angle_error = gt_heading - pre_heading
            if angle_error > 180:
                angle_error = angle_error - 360
            if angle_error < -180:
                angle_error = angle_error + 360
            error_angel.append(angle_error)
            error_distance.append(metri_gt - metri_est)
            temp_CEP=[]
            for j in range(1-min(10,len(gt)-i),min(10,len(gt)-i)-1):
                temp_CEP.append(np.sqrt(np.sum((est[i+j]-gt[i])**2)))
            CEP_distance.append(min(temp_CEP))
            ind.append(i)
            metri_gt = 0
            metri_est = 0
            temp = i
        if gt_sum_dis>50 and label_50==-1:
            label_50=i;
    # 计算50米左右100帧的距离误差
    if label_50==-1:
        label_50=len(gt)-100
    temp=0
    j=0
    for i in range(1-min(100,len(gt)-label_50),min(100,len(gt)-label_50)-1):
        # temp=abs(sum_dis[label_50][0]-sum_dis[label_50+i][1])
        # if temp<dis_error_50_offset:
        #     dis_error_50_offset=temp
        temp+=sum_dis[label_50][0]-sum_dis[label_50+i][1]
        j+=1
    dis_error_50_offset=temp/j

    for i in range(len(gt_headings) - 3):
        gt_headings_var.append(np.std(gt_headings[i:i + 3]))
    for i in range(len(est_headings) - 3):
        est_headings_var.append(np.std(est_headings[i:i + 3]))
    gt_headings_var.append(0)
    gt_headings_var.append(0)
    gt_headings_var.append(0)
    gt_headings_var = np.array(gt_headings_var)
    est_headings_var.append(0)
    est_headings_var.append(0)
    est_headings_var.append(0)
    est_headings_var = np.array(est_headings_var)
    wan = False
    chuwan = ruwan = 0
    for i in range(gt_headings_var.shape[0]):
        if gt_headings_var[i] >= 10:
            if not wan:
                ruwan = i
                ind_wan.append([chuwan + 1, ruwan + 1])
            wan = True
        else:
            if wan:
                chuwan = i
            wan = False
    for sub_ind in ind_wan:
        for i in range(sub_ind[0], sub_ind[1]):
            if abs(error_angel[i]) > 10:
                ind_15.append(i)
            error_angel_no_wan.append(error_angel[i])

    # wan = False
    # chuwan = ruwan = 0
    # for i in range(est_headings_var.shape[0]):
    #     if est_headings_var[i] >= 10:
    #         if not wan:
    #             ruwan = i
    #             ind_wan_est.append([chuwan + 1, ruwan + 1])
    #         wan = True
    #     else:
    #         if wan:
    #             chuwan = i
    #         wan = False
    # print("wan:",ind_wan,ind_wan_est)
    # for i in range(len(ind_wan_est)):
    #     duiqi.append([ind[ind_wan[i][0]],ind[ind_wan_est[i][0]]])
    #
    # duiqi=np.array(duiqi)

    return np.array(error_angel_no_wan), np.array(error_angel), np.array(ind_15), np.array(error_distance), \
           gt_headings_var, np.array(ind_wan), np.array(ind), np.array(sum_dis_error),np.array(ind_wan_est),\
           np.array(CEP_distance),dis_error_50_offset


def compute_50_rte(est, gt, duiqi):
    std_dis=60
    delta_dis=10
    sum_dis = []
    delta=[]
    error_rte_50 = []
    metri_gt = 0
    metri_est = 0
    CEP_68=0
    sum_dis.append([metri_gt, metri_est])
    for i in range(len(gt) - 1):
        metri_gt += np.sqrt((gt[i, 0] - gt[i + 1, 0]) ** 2 + (gt[i, 1] - gt[i + 1, 1]) ** 2)
        metri_est += np.sqrt((est[i, 0] - est[i + 1, 0]) ** 2 + (est[i, 1] - est[i + 1, 1]) ** 2)
        sum_dis.append([metri_gt, metri_est])
    if sum_dis[-1][0]<std_dis:
        std_dis=sum_dis[-10][0]
    for i in range(1):
        if sum_dis[i*20][0]  > delta_dis:
            break
        for j in range(i * 20 + 1, len(gt)):
            if sum_dis[j][0] - sum_dis[i*20][0] >= std_dis:
                for k in range(i*20,j+1):
                    for l in range(k+1,j+1):
                        if sum_dis[l][0]-sum_dis[k][0]>std_dis-delta_dis:
                            delta.append([k,l])
                            break
                rte_50m ,CEP_68= compute_relative_50_error(est, gt, delta,duiqi)
                # rte_50m = compute_relative_trajectory_error(est, gt, 1000)
                error_rte_50.append(rte_50m)
                delta=[]
                break
        break
    error_rte_50 = np.array(error_rte_50)
    rte = np.mean(error_rte_50)
    return rte, CEP_68


def get_heading(x1, x2):
    # 获取x1时刻的朝向，x2为x1后的数据。用于制作偏航角的标签
    return np.arctan2((x2[1] - x1[1]), (x2[0] - x1[0])) * 180 / np.pi


def get_serial_heading(pos_serial1, pos_serial2):
    headings = []
    for pos1, pos2 in zip(pos_serial1, pos_serial2):
        headings.append(get_heading(pos1, pos2))
    return headings
