import glob
import cv2
import re
import numpy as np
import torch

imgs_path = "E:\\project\\vehicle_dataset\\new_images\\test_data"


class_path = "E:\\project\\vehicle_dataset\\new_images\\test_data//classes.txt"

cfg_path = "E:\\project\\vehicle_dataset\\yolov3_model\\18000_epc\\yolov3.cfg"
weights_path = "E:\\project\\vehicle_dataset\\model_test\\yolov3_custom.weights"




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
   
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
  
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
