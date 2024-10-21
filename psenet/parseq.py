import torch 
import os
from PIL import Image,ImageOps,ImageDraw, ImageFont,ExifTags
import cv2
import cv2
import numpy as np 
import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import time
from datetime import datetime
import re
import torchvision.transforms as T
from PIL import Image, ImageDraw




class NeoOCRExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device",self.device)
        self.models_name = ['parseq']
        self._model_cache = {}
        self._preprocess = T.Compose([
            T.Resize((32, 128), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        # self._get_model(self.models_name)
        # Initialize models by iterating over the model names
        for model_name in self.models_name:
            self._get_model(model_name)  # Correctly load each model individually

    # def _get_model(self, names):
    #     for name in names:
    #         model = torch.hub.load('baudm/parseq', name, pretrained=True).eval().to(self.device)
    #         self._model_cache[name] = model
    #     return model

    def _get_model(self, name):
        if name in self._model_cache:
            return self._model_cache[name]
        try:
            # Load the model using torch.hub and cache it
            model = torch.hub.load('baudm/parseq', name, pretrained=True).eval().to(self.device)
            self._model_cache[name] = model
            return model
        except RuntimeError as e:
            print(f"Error loading model '{name}': {e}")
            raise


    def save_results_to_csv(self, results, csv_path):
        df = pd.DataFrame(results)
        
        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            # If it exists, append without writing the header
            df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"Results appended to {csv_path}")
        else:
            # If it doesn't exist, create a new file and write the header
            df.to_csv(csv_path, mode='w', index=False)
            print(f"Results saved to {csv_path}")


    # @staticmethod
    def draw_bounding_boxes(self, image, all_points, color=(0, 255, 0), thickness=2):
        for points in all_points:
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)
        
        return image

    # @staticmethod
    def generate_points_for_opencv(self, bounding_boxes):
        all_points = []
        # print(bounding_boxes)
        for bbox in bounding_boxes:
            # print(bbox)
            
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[1])
            x3, y3 = int(bbox[2]), int(bbox[3])
            x4, y4 = int(bbox[0]), int(bbox[3])
            
            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            
            all_points.append(points)
    
        return all_points

    # @staticmethod
    def get_rotate_crop_image(self, img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    
    # @staticmethod
    def get_minarea_rect_crop(self, img, points):
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    
        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2
    
        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img

    # @staticmethod
    def extract_selling_temp(self, text):
        # Regex pattern
        pattern = re.compile(
            r"(?i)(use\swithin|expires|shelf\s?life|valid\s?until|best\s?before|good\s?until|consume|recommended\s?use|use\sby)\s(?:\d+\s(?:years?|months?)\s(?:from|after|of|post(?:-| )?)\s(?:the\s)?(?:manufacture|production|manufacturing|manufacturing\sdate)?(?:date)?)|(?:(?:\d+\s(?:years?|months?))|(?:\d+\s(?:years?|months?)\s(?:after|from|of|post(?:-| )?)\s(?:the\s)?(?:manufacture|production|manufacturing|manufacturing\sdate)?(?:date)?))"
        )
        
        # Matching and printing results
        #for sentence in s:
        match = pattern.search(text)
        if match:
            return match.group(0)
        else:
            return None

    # @staticmethod
    def extract_mrp(self, ocr_text):
        # Define a regular expression pattern to match various MRP patterns
        mrp_pattern = re.compile(r'''
            (?:
                MRP\s*[:\-]?\s*          # Match "MRP" followed by optional colon or hyphen and whitespace
                (?:Rs\.?|₹|\u20ac)?         # Optionally match "Rs", "Rs.", "₹", or "€"
                (?:Rs\.?|₹)?                # Optionally match "Rs", "Rs.", or "₹"
                \s*                         # Optional whitespace
                [\*\$₹\-]?\s*               # Optionally match *, $, ₹, or -
                (\d{1,3}(?:,\d{3})*(?:\.\d{2})?)  # Capture the MRP amount (e.g., 179.00, 1,200.00)
                (?:/[\d\.]+)?               # Optionally match / followed by digits (e.g., 179.00/17.90)
            |
                Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)  # Match "Rs" followed by the amount
            |
                ₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)     # Match "₹" followed by the amount
            |
                (\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*MRP   # Match amount followed by "MRP"
            |
                (\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*/-    # Match amount followed by "/-" (e.g., 180/-)
            |
                \$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)        # Match "$" followed by the amount
            |
                \u20ac\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)  # Match "€" followed by the amount
            |
                \*\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)    # Match "*" followed by the amount
                 |
                (\d{2,4}\.00)          # Match 2 to 4 digits followed by ".00"
            )
            ''', re.VERBOSE | re.IGNORECASE)
    
        # Find all matches in the OCR text
        matches = mrp_pattern.findall(ocr_text)
        
        # Extract and clean MRP values
        mrp_values = []
        for match in matches:
            for value in match:
                if value:
                    # Remove extra characters and handle commas
                    cleaned_value = re.sub(r'[^\d\.]', '', value).replace(',', '')
                    if cleaned_value:
                        mrp_values.append(float(cleaned_value))
        
        # Return unique and sorted MRP values
        if mrp_values:
            return sorted(set(mrp_values))
        else:
            return None

    # @staticmethod
    def wrap_text(self, text, font, font_scale, font_thickness, max_width):
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if cv2.getTextSize(current_line + " " + word, font, font_scale, font_thickness)[0][0] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = current_line + " " + word if current_line else word
    
        lines.append(current_line)
        return lines

    # @staticmethod
    def extract_dates(self, text):
        # All accountable date patterns
        date_patterns = [
            # 12/11/2024 or 12.12.21
            r'\b(?:[0-9]|0[1-9]|[12][0-9]|3[01])\s*/\s*(?:0[1-9]|1[0-2])\s*/\s*(?:2[1-9]|202[0-9]|2030)\b',                                                 # DD/MM/YY or DD/MM/YYYY
            r'\b(?:0[1-9]|1[0-2])\s*/\s*(?:2[0-9]|202[0-9]|2030)\b',                                                                                        # MM/YY or MM/YYYY
            
            r'\b(?:[0-9]|0[1-9]|[12][0-9]|3[01])\s*-\s*(?:0[1-9]|1[0-2])\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',                                                 # DD-MM-YY or DD-MM-YYYY
            r'\b(?:0[1-9]|1[0-2])\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',                                                                                        # MM-YY or MM-YYYY
            
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*\.\s*(?:0[1-9]|1[0-2])\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',                                                     # DD.MM.YY or DD.MM.YYYY
            r'\b(?:0[1-9]|1[0-2])\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',                                                                                       # MM.YY or MM.YYYY
    
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*(?:0[1-9]|1[0-2])\s*(?:2[1-9]|202[0-9]|2030)\b',                                                               # DD MM YY or DD MM YYYY
            r'\b(?:0[1-9]|1[0-2])\s*(?:2[1-9]|202[0-9]|2030)\b',                                                                                            # MM YY or MM YYYY
    
            # 12/JAN/2024 or 12/JUN/2028
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(?:2[1-9]|202[0-9]|2030)\b',                             # DD MON YY or DD MON YYYY
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(?:2[1-9]|202[0-9]|2030)\b',                                                          # MON YY or MON YYYY
            
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*/\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*/\s*(?:2[1-9]|202[0-9]|2030)\b',                     # DD/MON/YY or DD/MON/YYYY
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*/\s*(?:2[1-9]|202[0-9]|2030)\b',                                                      # MON/YY or MON/YYYY
            
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*-\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',                     # DD-MON-YY or DD-MON-YYYY
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',                                                      # MON-YY or MON-YYYY
    
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*\.\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',                   # DD.MON.YY or DD.MON.YYYY
            r'\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',                                                     # MON.YY or MON.YYYY
    
            # 23-JANUARY-2026 or 19-JUNE-2022
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*(?:2[1-9]|202[0-9]|2030)\b',  # DD Month YYYY
            r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*(?:2[1-9]|202[0-9]|2030)\b',                               # Month YYYY
    
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*-\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',  # DD-Month-YYYY
            r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*-\s*(?:2[1-9]|202[0-9]|2030)\b',                                   # Month-YYYY
            
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*/\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*/\s*(?:2[1-9]|202[0-9]|2030)\b',  # DD/Month/YYYY
            r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*/\s*(?:2[1-9]|202[0-9]|2030)\b',                                   # Month/YYYY
            
            r'\b(?:0[1-9]|[12][0-9]|3[01])\s*\.\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',  # DD.Month.YYYY
            r'\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s*\.\s*(?:2[1-9]|202[0-9]|2030)\b',                                    # Month.YYYY    
        ]
        combined_pattern = '|'.join(date_patterns)
        matches = re.findall(combined_pattern, text)
        return matches

    # @staticmethod
    def validate_dates(self, dates):
        # valid_dates = []
        # for date in dates:
        #     date += '##'
        #     valid_dates.append(date)
    
        if len(dates)==2 and dates[0]==dates[1]:
            valid_dates = 'Mfg. Date:' + dates[0]
        elif len(dates)==2:
            valid_dates = 'Mfg. Date:' + dates[0] + ',  Exp. Date:' + dates[1]
        else:
            valid_dates = "No Date Detected" + "Mrp:"
            
        return valid_dates

    # @staticmethod
    def get_min_max_dates(self, date_strings):
        # Define date formats to try for parsing
        date_formats = [
            "%d/%m/%y","%d-%m-%y","%d.%m.%y","%d %m %y",  # Day/Month/Year (2-digit)
            "%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%d %m %Y",  # Day/Month/Year (4-digit)
            "%m/%Y","%m-%Y","%m.%Y","%m %Y",              # Month/Year (4-digit)
            "%m/%y","%m-%y","%m.%y","%m %y",              # Month/Year (2-digit)
    
            
            "%b/%y","%b-%y","%b.%y","%b %y","%b%y",     # Abbreviated Month/Year (e.g., Jan/21)
            "%b/%Y","%b-%Y","%b.%Y","%b %Y","%b%Y",     # Abbreviated Month/Year (e.g., Jan/2021)
            "%B/%Y","%B-%Y","%B.%Y","%B %Y","%B%Y",     # Full Month/Year (e.g., January 2021)
            "%B/%y","%B-%y","%B.%y","%B %y","%B%y",     # Full Month/Year (2-digit year, e.g., January 21)
            "%d/%b/%y","%d-%b-%y","%d.%b.%y","%d %b %y","%d%b%y",  # Day/Abbreviated Month/Year (e.g., 01 Jan 21)
            "%d/%b/%Y","%d-%b-%Y","%d.%b.%Y","%d %b %Y","%d%b%Y",  # Day/Abbreviated Month/Year (e.g., 01 Jan 2021) 
            "%d/%B/%y","%d-%B-%y","%d.%B.%y","%d %B %y","%d%B%y",  # Day/Abbreviated Month/Year (e.g., 01 Jan 21)
            "%d/%B/%Y","%d-%B-%Y","%d.%B.%Y","%d %B %Y","%d%B%Y",  # Day/Abbreviated Month/Year (e.g., 01 Jan 2021) 
        ]
        
        parsed_dates = []
        
        # Try to parse each date string using the formats
        for date_str in date_strings:
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    parsed_dates.append((parsed_date, date_str))  
                    break
                except ValueError:
                    continue
        
        if not parsed_dates:
            return []
    
        # Find the minimum and maximum dates
        min_date_str = min(parsed_dates)[1]
        max_date_str = max(parsed_dates)[1]
    
        return [min_date_str, max_date_str]
    
    def extract_batch_number(self, text):
        # Step 1: Look for BN No. type patterns
        bn_pattern = r'\b(?:BN\.?\s?(?:NO\.?)?\s?)([A-Z0-9]{3,10})\b'
        bn_match = re.search(bn_pattern, text, re.IGNORECASE)
        
        if bn_match:
            return bn_match.group(1)  # Return the captured batch number
        
        # Step 2: If no BN No. pattern found, look for alphanumeric patterns
        alphanumeric_patterns = [
            # Pattern for longer alphanumeric batch numbers (6-10 characters)
            r'\b(?:B|LOT)?[A-Z0-9]{6,10}\b',
            
            # Pattern for shorter batch numbers like GT147 or B037
            r'\b(?:B|LOT)?[A-Z]{1,2}[0-9]{3,4}\b'
        ]
        
        combined_pattern = '|'.join(alphanumeric_patterns)
        matches = re.findall(combined_pattern, text)
        
        # Filter matches to ensure they contain both uppercase letters and numbers
        valid_matches = [match for match in matches if re.search(r'\d', match) and re.search(r'[A-Z]', match)]
        
        # Return the longest valid match, or None if no valid matches
        return max(valid_matches, key=len) if valid_matches else None

    # @staticmethod
    # def add_text_to_image(self, image_path, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, font_thickness=2, margin=30):
    #     # Load the image
    #     image = cv2.imread(image_path)
    #     image_height, image_width = image.shape[:2]
    
    #     # Wrap the text to fit within the image width
    #     lines = self.wrap_text(text, font, font_scale, font_thickness, image_width - 2 * margin)
    
    #     # Calculate the height of the text
    #     line_height = cv2.getTextSize('Tg', font, font_scale, font_thickness)[0][1] + margin
    #     text_height = line_height * len(lines) + margin
    
    #     # Create a new image with a white background
    #     total_height = image_height + text_height
    #     canvas = np.ones((total_height, image_width, 3), dtype=np.uint8) * 255
    
    #     # Paste the original image onto the canvas
    #     canvas[0:image_height, 0:image_width] = image
    
    #     # Draw the text on the canvas
    #     y_text = image_height + margin
    #     for line in lines:
    #         text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
    #         text_x = margin
    #         text_y = y_text + text_size[1]
    #         cv2.putText(canvas, line, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    #         y_text += line_height
    
    #     # Save the final image
    #     return canvas
    #     # cv2.imwrite(output_path, canvas)

    # @staticmethod
    def add_space_and_plot(self, image_path, data_lists, space_height=200):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        height, width, channels = image.shape
        
        row_spacing = 50  # Space between each new row for ocr_map entries
        max_column_width = 300  # Maximum width of each column
        required_width = max(width, max_column_width + 100)  # Adjust width if necessary
        
        new_height = height + (len(data_lists) * (space_height + row_spacing))
        new_width = required_width
        new_image = np.ones((new_height, new_width, channels), dtype=np.uint8) * 255  # White background
        
        new_image[0:height, 0:width] = image
        
        pil_image = Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        font = ImageFont.truetype("Arial.ttf", 40)
        
        x, y = 10, height + 10  
        row_height = 40  # Height between each row of text
    
        ocr_map = {
            0: ["PARSEQ", (255, 0, 0)],
        }
    
        for i, data_list in enumerate(data_lists):
            draw.text((x, y), f"{ocr_map[i][0]}", fill=ocr_map[i][1], font=font)
            y += row_height
    
            for item in data_list:
                draw.text((x, y), item, fill=ocr_map[i][1], font=font)
                y += row_height  # Move to the next line
            
            y += row_spacing  # Add spacing after each OCR map entry
    
        final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return final_image

    # def _get_model(self, name):
    #     # Implementation

    @torch.inference_mode()
    def inf_model(self, model_name, images):
        if not images:
            return [], []  # Return empty results for empty input

        # print(len(images))
        model = self._get_model(model_name)

        images = [Image.fromarray(image) for image in images]
        images = torch.stack([self._preprocess(image.convert('RGB')) for image in images]).to(self.device)  

        preds = model(images).softmax(-1)

        labels = []
        raw_results = []

        for pred in preds:
            label, _ = model.tokenizer.decode(pred.unsqueeze(0))  
            max_len = 25 if model_name == 'crnn' else len(label[0]) + 1
            labels.append(label[0])

        return labels

    def get_ocr_results(self, model_name, image, bounding_box):
        if model_name not in self._model_cache:
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self._model_cache.keys())}")

        crops_lst = []

        # points = self.generate_points_for_opencv(bounding_box)
        # print(bounding_box)
        points=[]
        for bbox in bounding_box['bboxes']:
            # Convert each set of points to a numpy array of int32 type
            points.append(np.array(bbox, np.int32))
            # print(points)

        # points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        for point in points:
            # print(point)
            crops_lst.append( self.get_minarea_rect_crop(image, point))
    
        # ocr_model = batch_parseq()
        tic = time.time()
        ocr_results = self.inf_model(model_name, crops_lst)
        toc = time.time()
        print(toc-tic)
    
        # tic = time.time()
        # ocr_results = ocr_model(model_name, crops_lst)
        # toc = time.time()
    
    
        return ocr_results

    def process_text(self, text):
        dic = {          # List of image file names
            'Batch No': [],   
            'MRP': [],       # List of predicted MRP values
            'Mfg. Date': [],                 # List of extracted MRD values
            'Exp. Date': [],                 # List of extracted EXP values
        }
        
        batch_no = self.extract_batch_number(text)
    
        pharase = self.extract_mrp(text)
        price_list = pharase
    
        # Extract and validate dates from the text
        valid_dates = self.extract_dates(text.upper())
        valid_dates = self.get_min_max_dates(valid_dates)
        valid_dates_p = self.validate_dates(valid_dates)
    
        # Ensure there are at least two dates in the list
        if len(valid_dates) < 1:
            valid_dates.append('None')
        if len(valid_dates) < 2:
            valid_dates.append('None')
            valid_dates.append('None')
        
        # Ensure there's an MRP value
        if pharase is None:
            pharase = ['None']
        pharase = max(pharase)
        
        # Create output content string
        valid_dates_lst = valid_dates_p.split(",")
        valid_dates_lst = [date.lstrip().rstrip() for date in valid_dates_lst]
        
        combo = ['MRP: ' + str(pharase)]
        combo.extend(valid_dates_lst)
    
        # print(combo)
        # output_content = " ".join(combo)
    
        # Add extracted data to dictionary
        # dic['image_name'].append(filename + '.jpg')
        dic['Batch No'].append(batch_no)
        dic['MRP'].append(pharase)
        dic['Mfg. Date'].append(None if valid_dates[0] == 'None' else valid_dates[0])
        dic['Exp. Date'].append(None if valid_dates[1] == 'None' else valid_dates[1])
    
        # Convert list values to raw values (remove the list structure)
        dic = {key: (value[0] if value else None) for key, value in dic.items()}
    
        return dic, combo

    # @staticmethod
    def flatten_list(self, nested_list):
        return [item for sublist in nested_list for item in (self.flatten_list(sublist) if isinstance(sublist, list) else [sublist])]
    
    
    # @staticmethod
    def save_text_to_json(self, text, image_name, output_dir='.'):
        # Create the output file name based on the image name
        base_name, _ = os.path.splitext(image_name)
        json_file_name = f"{base_name}.json"
        json_file_path = os.path.join(output_dir, json_file_name)
        
        # Create a dictionary with the desired JSON format
        json_data = {
            "text": text
        }
        
        # Convert dictionary to JSON string
        json_string = json.dumps(json_data, indent=4)
        
        # Save the JSON string to a file
        with open(json_file_path, 'w') as file:
            file.write(json_string)
        
        print(f"JSON saved to {json_file_path}")


    def process_json_files(self, image, data):
        # os.makedirs(output_folder, exist_ok=True)
        state_dic = pd.DataFrame()

        # for filename in os.listdir(image_folder):
        #     if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more image formats if needed
        #         # try:
        #             # print(os.path.join(image_folder, filename.replace('.json', '.jpg')))
                
        #         image_path = os.path.join(image_folder, filename)
        #         image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize image (optional)
        # new_width = 720
        # new_height = 1280
        # resized_img = cv2.resize(image, (new_width, new_height))
        
        # pil_image = Image.fromarray(image)
        # output_path = "/home/neojetson/Projects/Main_PipeLine_Edge/psenet/"
        
        # # Generate a unique filename using timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # filename = f"image_with_boxes_{timestamp}.jpg"
        # full_path = os.path.join(output_path, filename)

        # Ensure the directory exists
        # os.makedirs(output_path, exist_ok=True)

        # # Save the image
        # pil_image.save(full_path)
        # print(f"Image with bounding boxes saved to: {full_path}")

        # image_file_name = filename.replace('.json', '.jpg')
        # image_file_path = os.path.join(image_folder, image_file_name)
        # file_path = os.path.join(input_folder, filename)

        # Define output file paths
        # output_filename = os.path.splitext(filename)[0] + '.txt'
        # output_image_file = os.path.splitext(filename)[0] + '.jpg'
        # img_output_path = os.path.join(output_folder, output_image_file)
        # image_file_path = os.path.join(output_folder, output_image_file)
        # output_dir_parseq = "/home/neojetson/Projects/parseq/parseq_jsons"
        parseq_results = self.get_ocr_results('parseq', image, data)
        # print("parseq result:",parseq_results)
        parseq_results_flattened = self.flatten_list(parseq_results)
        parseq_text = " ".join(str(item) for item in parseq_results_flattened)
        # print("parseq results: ", parseq_text)
        # self.save_text_to_json(parseq_text,filename,output_dir_parseq)
        
        parseq_dic, parseq_output_content = self.process_text(parseq_text)
        
        
        
        # output_image = self.add_space_and_plot(image_file_path, [ parseq_output_content ])
        print(parseq_dic)
        new_row = {
            'mrp_parseq': parseq_dic.get('MRP'),
            'mrd_parseq': parseq_dic.get('Mfg. Date'),
            'exp_parseq': parseq_dic.get('Exp. Date'),
        }
        # cv2.imwrite(img_output_path, output_image)
        state_dic = pd.concat([state_dic, pd.DataFrame([new_row])], ignore_index=True)
        # except:
        #     pass
            # break
        return parseq_dic

