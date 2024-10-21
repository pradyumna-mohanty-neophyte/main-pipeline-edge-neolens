from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import time
import cv2
import uvicorn
import asyncio
from PIL import Image
import httpx  # Import httpx for sending HTTP requests
# import socketio
# from fastapi_socketio import SocketManager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from main import CameraController  # Import your CameraController class
# from qwen_inference import ImageProcessor
# from ProductDetector import ProductDetector
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Initialize Socket.IO and mount it to the FastAPI app
# sio = socketio.AsyncServer(
#     async_mode='asgi',
#     cors_allowed_origins=['http://localhost:3000'],
#     )
# socket_manager = SocketManager(app=app, socketio_path='/ws/socket.io')
# app.mount('/ws', socket_manager)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origins you want to allow (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# qwen = ImageProcessor()
# Initialize CameraController object
camera = CameraController()
camera.initialize()
camera.set_exposure_time(47490)
camera.set_gain_value(11.0)

# Flag to check for new metadata
new_metadata_available = False
metadata_lock = threading.Lock()  # Lock for thread-safe operations
latest_metadata = None  # Variable to hold the latest metadata

# Create a background thread to continuously update frames
streaming_thread = threading.Thread(target=camera.stream, args=(30, 1), daemon=True)
streaming_thread.start()

# Initialize the Socket.IO client
# sio = socketio.AsyncClient()

# Define the connection URL to your Node.js server
NODE_SERVER_URL = "http://localhost:5000/api/metadata"

executor = ThreadPoolExecutor()

@app.get("/video")
async def video_feed():
    """Stream video frames in real-time using MJPEG format."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


async def generate_frames():
    """Generator function to capture frames and encode them to JPEG format."""
    while True:
        if camera.raw_image is not None:
            with camera.lock:
                # Convert the latest frame to BGR format (OpenCV)
                # image_bgr = cv2.cvtColor(camera.raw_image[438:1638, 944:2144], cv2.COLOR_RGB2BGR)
                image_bgr = cv2.cvtColor(camera.raw_image, cv2.COLOR_RGB2BGR)
                # Resize or perform any preprocessing as needed
                _, jpeg_frame = cv2.imencode('.jpg', image_bgr)
                frame = jpeg_frame.tobytes()

            # Yield frame in the format expected by MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Small delay to prevent CPU overload
        await asyncio.sleep(0.05)

    
if __name__ == "__main__":
    # Run FastAPI server with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)