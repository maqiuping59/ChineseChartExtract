import os
import pprint
from ultralytics.utils import ops
import torch
import random
from PIL import Image
import cv2
from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    Profile,
    check_img_size,
    increment_path,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import  smart_inference_mode
from sklearn.cluster import KMeans
import numpy as np
from paddleocr import PaddleOCR
import joblib
import pandas as pd



class ChartExtractor():
    def __init__(self,legend_detect_path="./weights/legend_detection/legned_detction.pt",
                 chart_det_path="./weights/chart_text_ocr/det/chart_det/",
                 chart_rec_path="./weights/rec/ch_PP-OCRv4_rec_server_infer",
                 legend_detect_data="./data/legend.yaml",
                 chart_text_classification_path="./weights/chart_text_classification/text_position_model.joblib",
                 chart_text_classification_scaler_path="./weights/chart_text_classification/text_position_scaler.joblib",
                 save_dir="./legend_detect",
                 conf_thres=0.25,
                 iou_thres=0.45,
                 imgsz=(640, 640),
                 output_dir="./output",
                 legend_det_path="./weights/det/ch_PP-OCRv4_det_infer",
                 legend_rec_path="./weights/rec/ch_PP-OCRv4_rec_server_infer",
                 device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),):
        self.legend_detect_path = legend_detect_path
        self.chart_det_path = chart_det_path
        self.chart_rec_path = chart_rec_path
        self.legend_detect_data = legend_detect_data
        self.legend_det_path = legend_det_path
        self.legend_rec_path = legend_rec_path
        self.save_dir = save_dir
        self.output_dir = output_dir
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device

        self.legend_ocr = PaddleOCR(use_angle_cls=True, rec_model_dir=self.legend_rec_path,
                    det_model_dir=self.legend_det_path,ocr_version="PP-OCRv3",savefile=True)
        
        self.chart_text_classification_model = joblib.load(chart_text_classification_path)
        self.chart_text_classification_scaler = joblib.load(chart_text_classification_scaler_path)

        self.chart_ocr = PaddleOCR(use_angle_cls=True, rec_model_dir=self.chart_rec_path,
                    det_model_dir=self.chart_det_path,ocr_version="PP-OCRv3",savefile=True)
        
    def save_one_box(self,xyxy, im, file=Path("im.jpg"), gain=1., pad=10, square=False, BGR=True, save=True):
        if not isinstance(xyxy, torch.Tensor):  # may be list
            xyxy = torch.stack(xyxy)
        b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
        if square:
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
        b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
        xyxy = ops.xywh2xyxy(b).long()
        xyxy = ops.clip_boxes(xyxy, im.shape)
        crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
        if save:
            cv2.imwrite(file, crop) 
        return crop
    
    @smart_inference_mode()
    def legend_detect(self,image_path):
        result = []

        model = DetectMultiBackend(self.legend_detect_path, device=self.device, dnn=False, data=self.legend_detect_data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)
        bs = 1  # batch_size

        dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, _, dt = 0, [], (Profile(device=self.device), Profile(device=self.device), Profile(device=self.device))
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=False, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=False, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=False, visualize=visualize)
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, max_det=1000)



            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
        
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
                imc = im0.copy() 
                annotator = Annotator(im0, line_width=3, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    count = 0
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"

                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        save_path = os.path.join(self.save_dir, f"{count}.jpg")
                        self.save_one_box(xyxy, imc, file=save_path, BGR=True)
                        result.append(save_path)
                        count += 1
        return result
    
    def extract_colors(self,image_path, n_colors=5, min_area_ratio=0.01):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
    
        
        legend_text = self.legend_ocr.ocr(image_path, cls=True)
        legend_text = legend_text[0]
        legend_text = [line[1][0] for line in legend_text]


        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        total_pixels = image.shape[0] * image.shape[1]
        pixels = blurred.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        labels = kmeans.fit_predict(pixels)
        colors = kmeans.cluster_centers_
        unique_labels, counts = np.unique(labels, return_counts=True)

        filtered_colors = []
        for i, color in enumerate(colors):
            ratio = counts[i] / total_pixels
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            if (ratio >= min_area_ratio and 
                hsv_color[1] > 50 and  
                30 < hsv_color[2] < 250):  
                filtered_colors.append(color)
        
        return legend_text[0],np.array(filtered_colors).astype(np.uint8)
    def segment_by_color(self,image_path, target_colors, tolerance=30):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        masks = []
        for color in target_colors:
            lower = np.array([max(0, c - tolerance) for c in color])
            upper = np.array([min(255, c + tolerance) for c in color])
            mask = cv2.inRange(image, lower, upper)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            masks.append(mask)
        
        return masks
    
    def find_bar_positions(self,mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 500:
                center_x = int(x + w/2)  
                center_y = int(y + h/2)  
                positions.append({
                    'x': center_x,
                    'y': center_y,
                    'width': int(w),
                    'height': int(h),
                    'area': int(area)
                })
        
        positions.sort(key=lambda p: p['y'])
        return positions
    
    def bar_mask(self,image_path,legend_path):
        legend_text,colors = self.extract_colors(legend_path, n_colors=5, min_area_ratio=0.05)
        if len(colors) == 0:
            return {}
            
        masks = self.segment_by_color(image_path, colors, tolerance=40)
        original_image = cv2.imread(image_path)
        color_positions = {}
        for i, mask in enumerate(masks):
            result = cv2.bitwise_and(original_image, original_image, mask=mask)
            output_path = os.path.join(self.output_dir, f"segment_{legend_text}_{i}.png")
            cv2.imwrite(output_path, result)
            mask_path = os.path.join(self.output_dir, f"mask_{legend_text}_{i}.png")
            cv2.imwrite(mask_path, mask)
            positions = self.find_bar_positions(mask)
            self.mask_result[legend_text] = positions
    
    def predict_single_position(self,features):
        features_scaled = self.chart_text_classification_scaler.transform(features)
        prediction = self.chart_text_classification_model.predict(features_scaled)
        probability = self.chart_text_classification_model.predict_proba(features_scaled)
        
        return prediction[0], probability[0]
    
    def xyxy2xywh(self,x):
        x_s = []
        y_s = []
        for i in range(len(x)):
            x_s.append(x[i][0])
            y_s.append(x[i][1])
        x_s = set(x_s)
        y_s = set(y_s)
        return int((min(x_s)+max(x_s))/2), int((min(y_s)+max(y_s))/2)
    
    def extract_features_from_labels(self,lines,w,h,is_veritcal=True):
        ytricks = []
        xtricks = []
        title = ""
        x_title = ""
        y_title = ""
        legend = []
        unit = ""
        if is_veritcal:
            direction = 1
        else:
            direction = 0
        for line in lines:
            text = line[1][0]
            text = text.replace(" ","")
            if "e" in text and (text.index("e") == 1 ):
                unit = text
            points = np.array([[direction,
                round(line[0][0][0]/w,4),round(line[0][0][1]/h,4),
                round(line[0][1][0]/w,4),round(line[0][1][1]/h,4),
                round(line[0][2][0]/w,4),round(line[0][2][1]/h,4),
                round(line[0][3][0]/w,4),round(line[0][3][1]/h,4)
            ]])
            pre_type,_ = self.predict_single_position(points)
            if pre_type == 0:
                title = text
            elif pre_type == 1:
                x_title = text
            elif pre_type == 2:
                y_title = text
            elif pre_type == 3:
                xtricks.append([text,line[0]])
            elif pre_type == 4:
                legend.append([text,line[0]])
            elif pre_type == 5:
                try:
                    ytricks.append([float(text),line[0]])
                except:
                    pass
            elif pre_type == 6:
                unit = text
        result = {
            "title":title,
            "x_title":x_title,
            "y_title":y_title,
            "xtricks":xtricks,
            "ytricks":ytricks,
            "legend":legend,
            "unit":unit
        }
        return result
    
    def check_bar_chart_type(self,result):
        labels = [[self.xyxy2xywh(line[0]), line[1][0]] for line in result]
        tricks = []
        for label in labels:
            txt = label[1].replace(" ","")
            if txt.isnumeric():
                tricks.append(label)
                
        sample_tricks = random.choices(tricks, k=2)
        if abs(sample_tricks[0][0][0]-sample_tricks[1][0][0]) > abs(sample_tricks[0][0][1]-sample_tricks[1][0][1]):
            is_veritcal = False
        else:
            is_veritcal = True
        return is_veritcal,tricks
    
    def find_bar_belong(self,labels,position):
        distance = {}
        for label in labels.keys():
            distance[label] = np.sqrt((labels[label][0]-position["x"])**2+(labels[label][1]-position["y"])**2)
        return min(distance,key=distance.get)
    
    def check_tricks(self,tricks, is_vertical):

        if is_vertical:

            sorted_tricks = sorted(tricks.items(), key=lambda x: x[1][1])
        else:
            sorted_tricks = sorted(tricks.items(), key=lambda x: x[1][0])
        
        cleaned_tricks = {}
        
        if len(sorted_tricks) > 0:
            cleaned_tricks[sorted_tricks[0][0]] = sorted_tricks[0][1]
        
        for i in range(len(sorted_tricks) - 1):
            current_value = sorted_tricks[i][0]
            next_value = sorted_tricks[i + 1][0]
            current_coords = sorted_tricks[i][1]
            next_coords = sorted_tricks[i + 1][1]
            
            if is_vertical:
                if current_value > next_value:
                    cleaned_tricks[next_value] = next_coords
            else:
                if current_value < next_value:
                    cleaned_tricks[next_value] = next_coords
        
        return cleaned_tricks
    
    def identify_bar_value_vertical(self,tricks,border_y):
        values = []
        tricks_list = list(tricks.items())  # 将dict_items转换为列表
        for _ in range(3):
            sample_tricks = random.sample(tricks_list, 2)
            tri1,tri2 = sample_tricks[0],sample_tricks[1]
            factor = (tri2[0] - tri1[0])/(tri2[1][1] - tri1[1][1])
            values.append(tri1[0] + factor*(border_y - tri1[1][1]))
        return round(sum(values)/len(values),2)

    def identify_bar_value_horizontal(self,tricks, border_x):
        values = []
        tricks_list = list(tricks.items())  # 将dict_items转换为列表
        for _ in range(3):
            sample_tricks = random.sample(tricks_list, 2)
            tri1,tri2 = sample_tricks[0],sample_tricks[1]
            factor = (tri2[0] - tri1[0])/(tri2[1][0] - tri1[1][0])
            values.append(tri1[0] + factor*(border_x - tri1[1][0]))
        return round(sum(values)/len(values),2) 

        
    def extract_chart(self, image_path):
        image = Image.open(image_path)
        w,h = image.size
        image.close()

        result = self.chart_ocr.ocr(image_path, cls=True)
        result = result[0]
        is_veritcal,tricks = self.check_bar_chart_type(result)
        ocr_result = self.extract_features_from_labels(result,w,h,is_veritcal)

        title = ocr_result["title"]
        x_title = ocr_result["x_title"]
        y_title = ocr_result["y_title"]
        xtricks = ocr_result["xtricks"]
        ytricks = ocr_result["ytricks"]
        unit = ocr_result["unit"]


        labels = {}
        tricks = {}
        for item in xtricks:
            labels[item[0]] = [int((item[1][0][0]+item[1][1][0])/2),(int(item[1][0][1]+item[1][2][1])/2)]
        for item in ytricks:
            tricks[item[0]] = [int((item[1][0][0]+item[1][1][0])/2),(int(item[1][0][1]+item[1][2][1])/2)]
        tricks = self.check_tricks(tricks,is_veritcal)
        legend_detect_result = self.legend_detect(image_path)
        self.mask_result = {}
        for legend_detect_result_path in legend_detect_result:
            self.bar_mask(image_path, legend_detect_result_path)

        table = {l:{} for l in labels.keys()}

        if is_veritcal:
            for legend in self.mask_result.keys():
                for bar in self.mask_result[legend]:
                    bottom_y = bar["y"] + bar["height"]//2
                    bottom_value = self.identify_bar_value_vertical(tricks,bottom_y)
                    top_y = bar["y"] - bar["height"]//2
                    top_value = self.identify_bar_value_vertical(tricks,top_y)
                    belong_label = self.find_bar_belong(labels,bar)
                    value = round(top_value - bottom_value,2)
                    table[belong_label][legend] = value
        else:
            for legend in self.mask_result.keys():
                for bar in self.mask_result[legend]:
                    left_x = bar["x"] - bar["width"]//2
                    left_value = self.identify_bar_value_horizontal(tricks,left_x)
                    right_x = bar["x"] + bar["width"]//2
                    right_value = self.identify_bar_value_horizontal(tricks,right_x)
                    belong_label = self.find_bar_belong(labels,bar)
                    value = round(right_value - left_value,2)
                    table[belong_label][legend] = value

        table = {k: v for k, v in table.items() if v}

        df = pd.DataFrame(table)
        df.to_csv(os.path.join(self.output_dir, "table.csv"), index=False)

        legned = self.mask_result.keys()
        table_str = "||"+"|".join(legned)+"|\n" +  "|--|"+"--|"*len(legned)+"\n"
        for label in table.keys():
            table_str += "|"+label+"|"
            for legend in legned:
                table_str += f"{table[label][legend]}|"
            table_str += "\n"
        print(table)
        return table_str
        

if __name__ == "__main__":
    chart_extractor = ChartExtractor()
    table = chart_extractor.extract_chart("./test.png")
    pprint.pprint(table)