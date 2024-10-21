import pyrealsense2 as rs
import cv2
import numpy as np

class RealSenseDepthProcessor:
    def __init__(self):
        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable depth stream
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start the pipeline
        self.pipeline.start(self.config)
        
        self.depth_frame = None
        self.previous_depth_image = None
    
    def update_frame(self):
        # Get frames from the RealSense pipeline
        frames = self.pipeline.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        
        if self.depth_frame:
            # Convert depth frame to numpy array
            depth_image = np.asanyarray(self.depth_frame.get_data())
            
            # Apply a smoothing filter (e.g., Gaussian blur) to reduce noise
            depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
            
            # Store the current depth image for future comparisons (optional)
            self.previous_depth_image = depth_image
    
    def draw_rectangle_and_center(self, x, y, width, height):
        if not self.depth_frame:
            raise RuntimeError("Frames are not available. Please call update_frame() first.")
        
        # Convert depth frame to numpy array
        depth_image = np.asanyarray(self.depth_frame.get_data())
        
      
        normalized_depth = np.uint8(depth_image)
        
        # Apply colormap
        color_mapped_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        
        # Draw the rectangle in red on the color-mapped image
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv2.rectangle(color_mapped_image, top_left, bottom_right, (0, 0, 255), 2)
        
        # Calculate the center of the rectangle
        center_x = x + width // 2
        center_y = y + height // 2
        cv2.circle(color_mapped_image, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Display the image
        # cv2.imshow('Depth Image with Jet Colormap and Rectangle', color_mapped_image)
        
        return center_x, center_y
    
    def distance_to_center(self, center_x, center_y):
        if not self.depth_frame:
            raise RuntimeError("Depth frame is not available. Please call update_frame() first.")
        
        depth_image = np.asanyarray(self.depth_frame.get_data())
        distance = depth_image[center_y, center_x] * 0.001  # Convert to meters
        
        return distance
    
    def average_distance_in_rectangle(self, x, y, width, height):
        if not self.depth_frame:
            raise RuntimeError("Depth frame is not available. Please call update_frame() first.")
        
        depth_image = np.asanyarray(self.depth_frame.get_data())
        rect_depth_values = depth_image[y:y+height, x:x+width]
        
        # Remove zeros (invalid depth values)
        valid_depth_values = rect_depth_values[rect_depth_values > 0]
        
        if valid_depth_values.size == 0:
            return 0
        
        average_distance = np.min(valid_depth_values) * 0.001  # Convert to meters
        
        return average_distance
    
    def stop(self):
        self.pipeline.stop()
        # cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    processor = RealSenseDepthProcessor()
    
    try:
        while True:
            processor.update_frame()
            center_x, center_y = processor.draw_rectangle_and_center(180, 100, 240, 280)
            
            distance = processor.distance_to_center(center_x, center_y)
            print(f"Distance to center: {distance:.2f} meters")
            
            avg_distance = processor.average_distance_in_rectangle(100, 100, 200, 200)
            print(f"Average distance in rectangle: {avg_distance:.2f} meters")
            
            # Break out of the loop to stop the script
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        processor.stop()

