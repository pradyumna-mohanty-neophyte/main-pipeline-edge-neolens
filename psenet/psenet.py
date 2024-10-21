
from psenet.models.builder import build_model

from psenet.models.utils.fuse_conv_bn import fuse_module
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from psenet.parseq import NeoOCRExtractor
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import time
import json
from datetime import datetime
from PIL import Image, ImageDraw

parseq = NeoOCRExtractor()

class pse_text_detector:
    def __init__(self,config):
        self.cfg = config
        self.max_size = config['inference']['max_size'] 
        self.min_size = config['inference']['min_size']
        detection_model = build_model(config['model'])
        # print(detection_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_model = self.load_model(detection_model,config["model_weight"]).eval().to(device)
        self.ocr_ext = parseq

    def load_model(self,model,weight_path):
        if weight_path is not None:
            if os.path.isfile(weight_path):
                print("Loading model and optimizer from checkpoint '{}'".format(weight_path))
                sys.stdout.flush()

                checkpoint = torch.load(weight_path,weights_only=True)

                d = dict()
                for key, value in checkpoint['state_dict'].items():
                    tmp = key[7:]
                    d[tmp] = value
                model.load_state_dict(d)
                model = fuse_module(model)
                # print("model",model)
                return model
            else:
                print("No checkpoint found at '{}'".format())
                raise
    
    def resize_image(self, image):
        h, w = image.shape[0:2]
        max_dim = max(h, w)
        scale = 1.0  # Default scale to 1.0 (no scaling)
        if max_dim > self.max_size:
            scale = self.max_size / max_dim
        elif max_dim < self.min_size:
            scale = self.min_size / max_dim
        img = self.scale_aligned(image, scale)
        return img
    
    def scale_aligned(self,img, scale):
        h, w = img.shape[0:2]
        h = int(h * scale + 0.5)
        w = int(w * scale + 0.5)
        if h % 32 != 0:
            h = h + (32 - h % 32)
        if w % 32 != 0:
            w = w + (32 - w % 32)
        img = cv2.resize(img, dsize=(w, h))
        return img

    def detect_text_on_image(self,image):
        img_meta = dict(
            img_size=np.array(image.shape[:2])
        )

        img = self.resize_image(image)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        data = dict(
            imgs=img,
            img_metas=img_meta
        )
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=self.cfg
        ))

        # forward
        with torch.no_grad():
            outputs = self.detection_model(**data)
        return outputs
    
    def draw_bounding_boxes(self, image, output):
        # Convert image from OpenCV format (BGR) to PIL format (RGB)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Iterate through the detected bounding boxes
        for bbox in output['bboxes']:
            bbox = bbox.reshape(-1, 2)  # Reshape the bbox to get (x, y) coordinate pairs
            points = bbox.astype(int)  # Ensure the points are integers
            # Draw the bounding box on the image
            points = [(point[0], point[1]) for point in points]  # Convert to list of tuples
            draw.line(points + [points[0]], fill=(0, 255, 0), width=2)

        # output_path = "/home/neojetson/Projects/Main_PipeLine_Edge/psenet/td_images/"
        
        # # Generate a unique filename using timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # filename = f"image_with_boxes_{timestamp}.jpg"
        # full_path = os.path.join(output_path, filename)

        # # Ensure the directory exists
        # os.makedirs(output_path, exist_ok=True)

        # # Save the image
        # pil_image.save(full_path)
        # print(f"Image with bounding boxes saved to: {full_path}")

    
    def process_and_save_images(self, image):
        # total_time = 0
        
        # for filename in os.listdir(input_folder):
        #     if filename.endswith(".jpg") or filename.endswith(".png"):
        #         image_path = os.path.join(input_folder, filename)
        #         image_cv = cv2.imread(image_path)
        #         resized_img = cv2.resize(image_cv, (720, 1280))

        #         # Detect text in the image
        #         st_time = time.time()
        # resized_img = cv2.resize(image, (720, 1280))
        output = self.detect_text_on_image(image)
        # print(output)
        # et_time = time.time()
        # per_time = et_time - st_time
        # total_time += per_time

        # Draw bounding boxes and save the image
        # image_with_boxes = self.draw_bounding_boxes(resized_img, output)
        # output_path = os.path.join(output_folder, filename)
        # cv2.imwrite(output_path, image_with_boxes)
        
        # Draw bounding boxes and save the image
        # self.draw_bounding_boxes(image, output)

        # Process OCR and save results
        bboxes_converted = [bbox.tolist() for bbox in output['bboxes']]
        bboxes_as_coordinates = [[(bbox[i], bbox[i+1]) for i in range(0, len(bbox), 2)] for bbox in bboxes_converted]
        all_detections = {'bboxes': bboxes_as_coordinates}

        # OCR Processing
        results = self.ocr_ext.process_json_files(image, all_detections)
        return results
        # print(results)
        # self.ocr_ext.save_results_to_csv(results, csv_path)

        # print(f"Processed and saved: {output_path}")

        # print(f"Total processing time: {total_time}")
    
# if __name__=="__main__":
#     config = {
#         "inference":{
#             'min_size':736,
#             'max_size':1024
#         },
#         "model_weight":"/home/neojetson/Projects/Main_PipeLine_Edge/psenet/checkpoint/checkpoint_20ep.pth.tar",
#         "model" : {
#             'type':'PSENet',
#             'backbone':{'type':'resnet50',
#                         'pretrained':True},
#             'neck':{
#                     'type':'FPN',
#                     'in_channels':(256, 512, 1024, 2048),
#                     'out_channels':128
#                     },
#             'detection_head':{
#                     'type':'PSENet_Head',
#                     'in_channels':1024,
#                     'hidden_dim':256,
#                     'num_classes':7,
#                     'loss_text':{
#                         'type':'DiceLoss',
#                         'loss_weight':0.7
#                     },
#                     'loss_kernel':{
#                         'type':'DiceLoss',
#                         'loss_weight':0.3
#                     }
#             }
#         },
#         'test_cfg': {
#         'kernel_num': 3 , # This value can be adjusted depending on the kernel settings you need
#         'min_area': 5,             # Minimum area of the text to be detected
#         'min_score': 0.5,
#         'bbox_type':'rect'
#     }
#     }
#     text_detector = pse_text_detector(config)
#     image_path = '/home/neojetson/Projects/saved_images/product_2.jpg'
#     csv_path = '/home/neojetson/Projects/Main_PipeLine_Edge/results.csv'
#     image = cv2.imread(image_path)

#     text_detector.process_and_save_images(image, csv_path)




