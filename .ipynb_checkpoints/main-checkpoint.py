import os
import ids_peak.ids_peak as ids_peak
import ids_peak_afl.ids_peak_afl as ids_peak_afl
import ids_peak_ipl.ids_peak_ipl as ids_peak_ipl
import ids_peak.ids_peak_ipl_extension as ids_ipl_extension
import numpy as np
import pandas as pd
import cv2
import time
import threading
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from psenet.parseq import NeoOCRExtractor
from psenet.psenet import pse_text_detector
from pynput import keyboard  # Importing pynput for keyboard control
from matplotlib import pyplot as plt
from realsensetest import RealSenseDepthProcessor
# from psenet.psenet import pse_text_detector
# from qwen_inference import ImageProcessor

# Initialize RealSenseDepthProcessor
config = {
        "inference":{
            'min_size':736,
            'max_size':1024
        },
        "model_weight":"/home/neojetson/Projects/Main_PipeLine_Edge/psenet/checkpoint/checkpoint_20ep.pth.tar",
        "model" : {
            'type':'PSENet',
            'backbone':{'type':'resnet50',
                        'pretrained':True},
            'neck':{
                    'type':'FPN',
                    'in_channels':(256, 512, 1024, 2048),
                    'out_channels':128
                    },
            'detection_head':{
                    'type':'PSENet_Head',
                    'in_channels':1024,
                    'hidden_dim':256,
                    'num_classes':7,
                    'loss_text':{
                        'type':'DiceLoss',
                        'loss_weight':0.7
                    },
                    'loss_kernel':{
                        'type':'DiceLoss',
                        'loss_weight':0.3
                    }
            }
        },
        'test_cfg': {
        'kernel_num': 3 , # This value can be adjusted depending on the kernel settings you need
        'min_area': 5,             # Minimum area of the text to be detected
        'min_score': 0.5,
        'bbox_type':'rect'
    }
    }
rs = RealSenseDepthProcessor()
edge = pse_text_detector(config)
# qwen = ImageProcessor()

class CameraController:
    def __init__(self):
        self.device = None
        self.datastream = None
        self.remote_device_nodemap = None
        self.manager = None
        self.raw_image = None  # Shared image between threads
        self.lock = threading.Lock()  # Lock to synchronize access
        self.df = pd.DataFrame(columns=["MRP", "MFG", "EXP"])

    def initialize(self):
        ids_peak.Library.Initialize()
        ids_peak_afl.Library.Init()
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        device_descriptors = device_manager.Devices()
        
        if len(device_descriptors) == 0:
            raise Exception("No devices found")
        
        self.device = device_descriptors[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        print(f"Opened Device: {self.device.DisplayName()}")
        
        self.remote_device_nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self._setup_datastream()
       
    def _setup_datastream(self):
        self.datastream = self.device.DataStreams()[0].OpenDataStream()
        payload_size = self.remote_device_nodemap.FindNode("PayloadSize").Value()
        for _ in range(self.datastream.NumBuffersAnnouncedMinRequired()):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.datastream.QueueBuffer(buffer)
        
        self.datastream.StartAcquisition()
        self.remote_device_nodemap.FindNode("AcquisitionStart").Execute()
        self.remote_device_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

    def set_exposure_time(self, exposure_time_us):
        self.remote_device_nodemap.FindNode("ExposureTime").SetValue(exposure_time_us)

    def set_focus_value(self, focus_value):
        self.remote_device_nodemap.FindNode("FocusStepper").SetValue(int(focus_value))
        
    def set_gain_value(self, gain_value):
        self.remote_device_nodemap.FindNode("Gain").SetValue(float(gain_value))
        
    def get_stepper_value(self, distance):
        table = [
            (50.6, 700), (55.1, 707), (60.2, 713), (65.0, 718), (70.4, 723),
            (73.4, 726), (76.1, 724), (80.3, 726), (85.1, 730), (90.2, 732),
            (95.0, 735), (100.1, 738), (105.4, 740), (110.2, 741), (115.1, 743),
            (120.2, 745), (124.8, 749), (126.6, 747), (130.3, 751), (134.1, 752),
            (137.4, 751), (140.2, 753), (145.0, 754), (147.5, 749), (150.3, 750),
            (153.2, 751), (155.8, 751), (159.7, 752), (164.2, 753), (166.5, 752),
            (169.8, 753), (173.5, 754)
        ]
        for i in range(len(table) - 1):
            lower_range, lower_stepper = table[i]
            upper_range, upper_stepper = table[i + 1]
            if lower_range <= distance < upper_range:
                delta_distance = distance - lower_range
                delta_stepper = upper_stepper - lower_stepper
                delta_range = upper_range - lower_range
                return lower_stepper + (delta_stepper / delta_range) * delta_distance
        return table[-1][1]

    def capture_image(self):
        try:
            buffer = self.datastream.WaitForFinishedBuffer(1000)
            raw_image = ids_ipl_extension.BufferToImage(buffer)
            color_image = raw_image.ConvertTo(ids_peak_ipl.PixelFormatName_RGB8)
            self.datastream.QueueBuffer(buffer)

            image_np_array = color_image.get_numpy_3D()
            with self.lock:
                self.raw_image = image_np_array.copy()

        except Exception as e:
            print("Exception: ", e)
            return None

    def stream(self, fps=20, skip_frames=0):
        """
        Stream images from the camera.
        
        :param fps: Desired frames per second
        :param skip_frames: Number of frames to skip between each captured frame
        """
        frame_time = 1 / fps
        frame_count = 0

        try:
            while True:
                start_time = time.time()

                # Capture and process image
                self.capture_image()
                
                if self.raw_image is not None:
                    frame_count += 1

                    # Only process and display every (skip_frames + 1)th frame
                    if frame_count % (skip_frames + 1) == 0:
                        # Convert to BGR for OpenCV
                        with self.lock:
                            image_bgr = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2BGR)
                        image_resized = cv2.resize(image_bgr, (1280, 720))
                        
                        # Display the image in the resizable window
                        cv2.imshow('Camera Stream', image_resized)
                        
                        # Update RealSense frame
                        rs.update_frame()
                        
                        # Get and process distance values
                        center_x, center_y = rs.draw_rectangle_and_center(180, 100, 240, 280)
                        avg_distance = rs.average_distance_in_rectangle(100, 100, 200, 200)
                        
                        self.set_focus_value(int(self.get_stepper_value(avg_distance * 100)))
                
                # Calculate sleep time to maintain desired FPS
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed_time)
                time.sleep(sleep_time)

                # Check if 'q' is pressed to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting stream...")
                    break

        except KeyboardInterrupt:
            print("Streaming stopped by user.")
        finally:
            cv2.destroyAllWindows()
            rs.stop()

    def add_to_dataframe(self, parseq_dic, df=None):
        # Create a new row as a dictionary
        new_row = {
            'MRP': parseq_dic.get('MRP'),
            'MFG': parseq_dic.get('MRD'),
            'EXP': parseq_dic.get('EXP'),
        }
        
        # Use pd.concat to append the new row to the DataFrame
        new_row_df = pd.DataFrame([new_row])  # Create a new DataFrame with a single row
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)  # Concatenate with the existing DataFrame

        return df  # Return the updated DataFrame

    def on_press(self, key):
        """Handle key presses to process metadata on 'c' press."""
        try:
            if key.char == 'c':
                with self.lock:
                    if self.raw_image is not None:
                        # metadata = qwen.process_image_from_array(self.raw_image)
                        metadata = edge.process_and_save_images(self.raw_image)
                        self.add_to_dataframe(metadata)
                        print(self.df)
                        if self.df is not None and not self.df.empty:
                            csv_save_path = "/home/neojetson/Projects/Main_PipeLine_Edge/results.csv"
                            self.df.to_csv(csv_save_path, index=False)
                            print(f"CSV file saved successfully at: {csv_save_path}")
                        else:
                            print("DataFrame is empty or not defined. Cannot save CSV.")
                        if metadata:
                            print("Metadata:", metadata)
                        else:
                            print("Failed to retrieve metadata")
            # elif key.char == 's':
            #     with self.lock:
            #         if self.df is not None and not self.df.empty:
            #             csv_save_path = "/home/neojetson/Projects/Main_PipeLine_Edge/results.csv"
            #             self.df.to_csv(csv_save_path, index=False)
            #             print(f"CSV file saved successfully at: {csv_save_path}")
            #         else:
            #             print("DataFrame is empty or not defined. Cannot save CSV.")
        except AttributeError:
            pass

# Usage example:
if __name__ == "__main__":
    camera = CameraController()
    camera.initialize()
    camera.set_exposure_time(37490) 
    camera.set_gain_value(8.5)

    # Start the stream thread
    streaming_thread = threading.Thread(target=camera.stream, args=(30, 1))
    streaming_thread.start()

    # Start the keyboard listener in the main thread
    with keyboard.Listener(on_press=camera.on_press) as listener:
        listener.join()

