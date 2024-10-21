import cv2
from ultralytics import YOLO


class ProductDetector:

    def __init__(self):

        self.body_model_path = '/home/neojetson/Projects/Main_PipeLine_Edge/best.pt'
        self.body_det_threshold = 0.7

        self.roi_point = [374, 12, 1840, 1291]
        self.roi_threshold = 10

        self._load_models()
        

    def _load_models(self):

        """
        Private method to load machine learning models required for body detection tasks.

        This method initializes and loads the YOLO (You Only Look Once) model for body detection.
        The model is loaded from the specified file path (`self.body_model_path`) and assigned
        to the `self.body_model` attribute for further use in the application.

        Parameters:
        None

        Returns:
        None

        Side Effects:
        - Loads the YOLO model from the provided path and assigns it to `self.body_model`.
        """

        self.body_model = YOLO(self.body_model_path)



    def _find_person_in_roi(self, bbox):

        """
        Check if bbox1 is inside bbox2.
        
        Parameters:
        bbox: Tuple[int, int, int, int] - Coordinates of the human bounding box (x1, y1, x2, y2).

        Returns:
        bool: True if human bbox is inside roi, otherwise False.
        """
        
        # Unpack coordinates for human bbox and roi
        x1_1, y1_1, x2_1, y2_1 = bbox
        x1_2, y1_2, x2_2, y2_2 = self.roi_point
        
        # Check for overlap
        if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
            return True

        # Check if they are close horizontally
        if abs(x1_1 - x2_2) <= self.roi_threshold or abs(x2_1 - x1_2) <= self.roi_threshold:
            return True

        # Check if they are close vertically
        if abs(y1_1 - y2_2) <= self.roi_threshold or abs(y2_1 - y1_2) <= self.roi_threshold:
            return True

        return False



    def process(self, frame):


        bbox_list = []
        track_id_list = []
        confidence_list = []

        results = self.body_model.track(frame, conf=self.body_det_threshold, persist=True, verbose=False, device=0)

        if results[0].boxes.id is not None:
            bboxes = results[0].boxes.xyxy.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            for bbox, track, conf in zip(bboxes, track_ids, confidences):
                # if self._find_person_in_roi(bbox):
                bbox_list.append(bbox)
                track_id_list.append(track)
                confidence_list.append(conf)

                # Draw bounding box
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # Display track ID and confidence
                label = f'ID: {track}, Conf: {conf:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # cv2.rectangle(frame, (bbox[0], bbox[1] - 20), (bbox[0] + w, bbox[1]), (0, 255, 0), -1)
                # cv2.putText(frame, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


        processed_data = {
            "bboxes": bbox_list,
            "track_ids": track_id_list,
            "confidences": confidence_list
        }

        # print(f'body: {bbox_list}')

        # return frame
        return processed_data
        

