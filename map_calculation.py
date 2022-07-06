import glob
import cv2
import re
import numpy as np
import torch

imgs_path = "E:\\master_thesis\\project\\vehicle_dataset\\new_images\\test_data"


class_path = "E:\\master_thesis\\project\\vehicle_dataset\\new_images\\test_data//classes.txt"

cfg_path = "E:\\master_thesis\\project\\vehicle_dataset\\yolov3_model\\18000_epc\\yolov3.cfg"
weights_path = "E:\\master_thesis\\project\\vehicle_dataset\\model_test\\yolov3_custom_last1.weights"


#weights_path = "E:\\master_thesis\\project\\vehicle_dataset\\yolov3_model\\yolov3_final.weights"

#weights_path = "E:\\master_thesis\\project\\vehicle_dataset\\model_test\\yolov3_custom_last1.weights"

def create_boxes(imgs_path, class_path):
    true_boxes = []
    classes_path = open(class_path)
    img_path = glob.glob(imgs_path+"//*.jpg")
    class_strng = classes_path.readlines()
    classes = [i.strip() for i in class_strng]
    for i in img_path:

        img = cv2.imread(i, cv2.IMREAD_UNCHANGED)

        width = int(img.shape[1])
        height = int(img.shape[0])

        img_name = i[62:]
        img_name = img_name[:-4]
        file_name = i[:-4] + ".txt"
        file = open(file_name, "r")
        lines = file.readlines()

        for j in lines:
            boxes = []
            clas = classes[int(j[0])]
            yolo_array = re.split("\s", j.rstrip())

            x_yolo = float(yolo_array[1])
            y_yolo = float(yolo_array[2])
            yolo_width = float(yolo_array[3])
            yolo_height = float(yolo_array[4])

            box_width = yolo_width * width
            box_height = yolo_height * height
            x1 = str(int(x_yolo * width - (box_width / 2)))
            y1 = str(int(y_yolo * height - (box_height / 2)))
            x2 = str(int(x_yolo * width + (box_width / 2)))
            y2 = str(int(y_yolo * height + (box_height / 2)))
            boxes.append(int(img_name))
            boxes.append(clas)
            boxes.append(float(1))
            boxes.append(int(x1))
            boxes.append(int(y1))
            boxes.append(int(x2))
            boxes.append(int(y2))
            true_boxes.append(boxes)
    return true_boxes


true_boxes = create_boxes(imgs_path,class_path)

print("Ground truth boxes",true_boxes)


def create_pred_boxes(img_folder_path, cfg_path, weights_path,classes_txt_path):
    pred_boxes = []
    img_path = glob.glob(img_folder_path + "//*.jpg")
    classes_path = open(classes_txt_path)
    class_strng = classes_path.readlines()
    classes = [i.strip() for i in class_strng]
    for i in img_path:

        img_name = i[62:]
        img_name = img_name[:-4]
        img1 = cv2.imread(i, cv2.IMREAD_UNCHANGED)
        ht, wt, _ = img1.shape
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        blob = cv2.dnn.blobFromImage(img1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        last_layer = net.getUnconnectedOutLayersNames()
        layer_out = net.forward(last_layer)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_out:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > .6:
                    center_x = int(detection[0] * wt)
                    center_y = int(detection[1] * ht)
                    w = int(detection[2] * wt)
                    h = int(detection[3] * ht)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
        if len(indexes) > 0:
            for k in indexes.flatten():
                n_boxes = []
                x, y, w, h = boxes[k]
                confidence = round(confidences[k], 2)
                label = str(classes[class_ids[k]])
                n_boxes.append(int(img_name))
                n_boxes.append(label)
                n_boxes.append(confidence)
                n_boxes.append(x)
                n_boxes.append(y)
                n_boxes.append(w)
                n_boxes.append(h)
                pred_boxes.append(n_boxes)
    return pred_boxes


prediction_boxes = create_pred_boxes(imgs_path,cfg_path,weights_path,class_path)

print("predicted boxes",prediction_boxes)


#true_boxes = [[0, 'car', 1.0, 74, 36, 390, 238], [1, 'car', 1.0, 81, 132, 686, 504], [10, 'motor_bike', 1.0, 31, 20, 227, 134], [11, 'motor_bike', 1.0, 31, 19, 230, 130], [12, 'motor_bike', 1.0, 35, 26, 228, 144], [13, 'motor_bike', 1.0, 32, 23, 229, 134], [14, 'motor_bike', 1.0, 35, 25, 230, 147], [15, 'motor_bike', 1.0, 36, 24, 231, 141], [16, 'motor_bike', 1.0, 40, 26, 222, 134], [17, 'motor_bike', 1.0, 34, 27, 227, 128], [18, 'motor_bike', 1.0, 42, 31, 224, 131], [19, 'motor_bike', 1.0, 48, 43, 207, 128], [2, 'car', 1.0, 18, 34, 622, 371], [20, 'bus', 1.0, 20, 31, 226, 162], [21, 'bus', 1.0, 42, 15, 228, 178], [22, 'truck', 1.0, 59, 20, 235, 148], [23, 'truck', 1.0, 1, 54, 232, 202], [23, 'pedestrian', 1.0, 232, 114, 256, 177], [24, 'bus', 1.0, 27, 42, 215, 147], [25, 'truck', 1.0, 55, 31, 201, 133], [25, 'car', 1.0, 1, 78, 69, 128], [25, 'car', 1.0, 214, 81, 256, 107], [26, 'truck', 1.0, 59, 23, 245, 154], [27, 'bus', 1.0, 42, 39, 220, 171], [28, 'truck', 1.0, 132, 53, 251, 157], [28, 'truck', 1.0, 24, 57, 138, 157], [29, 'bus', 1.0, 37, 0, 244, 159], [3, 'car', 1.0, 59, 52, 454, 265], [30, 'truck', 1.0, 15, 59, 246, 137], [31, 'truck', 1.0, 16, 26, 207, 171], [32, 'bus', 1.0, 26, 25, 228, 113], [33, 'bus', 1.0, 32, 14, 239, 144], [34, 'bus', 1.0, 70, 9, 231, 164], [35, 'truck', 1.0, 52, 42, 228, 157], [36, 'bicycle', 1.0, 12, 0, 632, 354], [37, 'bicycle', 1.0, 38, 23, 602, 359], [38, 'bicycle', 1.0, 2, 13, 448, 283], [39, 'bicycle', 1.0, 98, 135, 493, 384], [4, 'car', 1.0, 22, 81, 593, 369], [40, 'bicycle', 1.0, 7, 5, 210, 159], [41, 'bicycle', 1.0, 72, 12, 596, 369], [42, 'bicycle', 1.0, 99, 123, 374, 308], [42, 'pedestrian', 1.0, 181, 29, 264, 286], [43, 'bicycle', 1.0, 10, 17, 592, 393], [44, 'bicycle', 1.0, 75, 188, 563, 591], [45, 'bicycle', 1.0, 9, 30, 639, 389], [46, 'bus', 1.0, 57, 3, 248, 183], [47, 'truck', 1.0, 19, 21, 249, 156], [48, 'truck', 1.0, 54, 22, 225, 170], [49, 'bus', 1.0, 0, 57, 256, 147], [5, 'car', 1.0, 150, 440, 1086, 808], [50, 'bus', 1.0, 55, 28, 192, 152], [53, 'pedestrian', 1.0, 142, 104, 214, 320], [53, 'pedestrian', 1.0, 218, 59, 302, 316], [53, 'pedestrian', 1.0, 321, 68, 445, 329], [54, 'pedestrian', 1.0, 160, 71, 234, 336], [54, 'pedestrian', 1.0, 217, 88, 314, 335], [55, 'pedestrian', 1.0, 254, 126, 297, 260], [55, 'pedestrian', 1.0, 205, 123, 245, 252], [55, 'pedestrian', 1.0, 168, 112, 203, 251], [56, 'pedestrian', 1.0, 105, 44, 259, 375], [57, 'pedestrian', 1.0, 144, 45, 305, 469], [58, 'pedestrian', 1.0, 60, 52, 214, 298], [58, 'pedestrian', 1.0, 205, 43, 355, 299], [59, 'pedestrian', 1.0, 97, 1, 325, 499], [6, 'car', 1.0, 158, 258, 506, 393], [60, 'pedestrian', 1.0, 229, 166, 420, 373], [7, 'car', 1.0, 136, 83, 962, 643], [8, 'car', 1.0, 57, 114, 930, 617], [9, 'car', 1.0, 199, 197, 1970, 1384]]

#pred_boxes = [[0, 'car', 0.99, 78, 32, 310, 203], [1, 'car', 1.0, 93, 146, 575, 357], [10, 'motor_bike', 1.0, 34, 12, 194, 124], [11, 'motor_bike', 1.0, 30, 11, 197, 122], [12, 'motor_bike', 1.0, 37, 27, 192, 119], [13, 'motor_bike', 1.0, 40, 19, 184, 114], [14, 'motor_bike', 1.0, 41, 24, 186, 124], [15, 'motor_bike', 0.99, 41, 21, 188, 117], [16, 'motor_bike', 1.0, 46, 21, 173, 113], [17, 'motor_bike', 1.0, 43, 24, 181, 105], [18, 'motor_bike', 1.0, 34, 27, 188, 103], [19, 'bicycle', 0.77, 49, 43, 157, 92], [2, 'car', 1.0, 5, 30, 635, 341], [20, 'bus', 1.0, 13, 18, 218, 153], [21, 'bus', 1.0, 39, 19, 192, 154], [22, 'truck', 1.0, 49, 18, 195, 126], [23, 'truck', 0.98, 0, 45, 255, 150], [23, 'pedestrian', 0.96, 231, 110, 23, 69], [24, 'bus', 1.0, 27, 41, 190, 109], [24, 'bus', 0.71, 222, 61, 31, 61], [25, 'truck', 0.99, 57, 25, 146, 105], [26, 'truck', 1.0, 60, 22, 186, 138], [27, 'bus', 1.0, 40, 40, 186, 129], [28, 'truck', 0.97, 137, 53, 107, 101], [29, 'bus', 0.99, 31, 2, 223, 156], [3, 'car', 1.0, 55, 57, 401, 198], [30, 'truck', 1.0, 9, 60, 233, 71], [31, 'truck', 1.0, 10, 25, 204, 145], [32, 'bus', 1.0, 18, 23, 219, 91], [33, 'bus', 1.0, 22, 18, 223, 128], [34, 'bus', 1.0, 74, 3, 156, 165], [34, 'bus', 0.97, -3, 39, 76, 81], [35, 'truck', 1.0, 52, 37, 178, 126], [36, 'bicycle', 1.0, 10, 0, 629, 353], [37, 'bicycle', 1.0, 40, 18, 556, 342], [38, 'bicycle', 1.0, -8, 21, 469, 265], [39, 'bicycle', 0.92, 108, 122, 368, 265], [4, 'car', 0.97, 9, 75, 599, 297], [40, 'bicycle', 1.0, 10, 12, 205, 141], [41, 'bicycle', 1.0, 71, 23, 533, 347], [42, 'bicycle', 0.99, 98, 140, 283, 173], [43, 'bicycle', 1.0, 10, 21, 593, 374], [44, 'bicycle', 1.0, 43, 69, 518, 526], [45, 'bicycle', 1.0, 5, 31, 640, 364], [46, 'bus', 0.99, 46, 8, 204, 180], [47, 'truck', 1.0, 10, 22, 247, 130], [48, 'truck', 1.0, 56, 20, 159, 149], [49, 'bus', 1.0, 5, 12, 243, 145], [5, 'car', 1.0, 98, 442, 1031, 382], [50, 'bus', 1.0, 58, 30, 129, 125], [53, 'car', 0.82, -11, 117, 189, 149], [53, 'pedestrian', 0.79, 143, 129, 74, 162], [55, 'bus', 0.89, 36, 4, 429, 194], [56, 'car', 0.92, 385, 140, 110, 100], [56, 'pedestrian', 0.89, 117, 61, 160, 318], [57, 'pedestrian', 0.87, 137, 54, 153, 396], [6, 'car', 1.0, 151, 258, 360, 132], [7, 'car', 0.98, 142, 66, 797, 559], [8, 'car', 0.99, 32, 99, 891, 518], [9, 'car', 1.0, 133, 261, 1842, 1071]]


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[3]
    box1_y1 = boxes_preds[4]
    box1_x2 = boxes_preds[5]
    box1_y2 = boxes_preds[6]
    box2_x1 = boxes_labels[3]
    box2_y1 = boxes_labels[4]
    box2_x2 = boxes_labels[5]
    box2_y2 = boxes_labels[6]

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)






def mAp(pred_boxes, true_boxes, iou_threshold=0.5, classes=[]):
    AP = []
    epsilon = 1e-6
    for c in classes:
        detections = []
        ground_truths = []

        for d in pred_boxes:
            if d[1] == c:
                detections.append(d)
        for t in true_boxes:
            if t[1] == c:
                ground_truths.append(t)
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = [0 for i in range(len(detections))]
        FP = [0 for i in range(len(detections))]

        total_true_bboxes = len(ground_truths)

        detected_box = []

        for idx, db in enumerate(detections):
            best_iou = 0
            best_box = []
            for tb in ground_truths:
                if db[0] == tb[0]:
                    iou = intersection_over_union(db, tb)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = tb
            if best_iou > iou_threshold and best_box not in detected_box:
                TP[idx] = 1
                detected_box.append(best_box)
            else:
                FP[idx] = 1
        TP_cumsum = torch.cumsum(torch.tensor(TP), dim=0)
        FP_cumsum = torch.cumsum(torch.tensor(FP), dim=0)
        # print("class:",c,"\n","TP:",TP,"\n","FP:",FP,"\n","TP_C:",TP_cumsum,"\n","FP_C:",FP_cumsum)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # print("recalls:::",recalls,"\n","precisions:",precisions)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        #fig, ax = plt.subplots()

        #ax.step(precisions, recalls)

        #plt.show()

        # torch.trapz for numerical integration
        AP.append(torch.trapz(precisions, recalls))

    # print(average_precisions)

    return sum(AP) / len(AP)

    # print(detections,"\n","g",ground_truths)


map = mAp(prediction_boxes, true_boxes, iou_threshold=0.5, classes=["car", "truck", "bicycle","pedestrian","bus","motor_bike"])
print(map)