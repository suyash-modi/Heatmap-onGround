# Product Analytics + Multi-Camera Person Tracking System

A Flask-based product analytics system that uses OpenVINO for person detection and re-identification across multiple camera feeds. Tracks product zone interactions, generates heatmaps, and provides real-time analytics through a web dashboard.

## Features

- **Multi-Camera Person Detection**: Uses OpenVINO person-detection-retail-0013 model for real-time detection
- **Person Re-Identification**: Tracks persons across multiple cameras using retail-0287 ReID model
- **Product Zone Analytics**: Tracks footfall and dwell time for product zones from MongoDB
- **Heatmap Visualization**: Real-time heatmap overlay showing person movement patterns
- **Path Tracing**: Visualizes person trajectories on video feeds
- **Live Dashboard**: Web-based UI displaying product analytics and live video feeds
- **MongoDB Integration**: Stores zones, footfall data, and hourly/daily statistics
- **Background Workers**: Automatic hourly and daily statistics aggregation

## Requirements

- Python 3.10+
- OpenVINO 2024.6+
- MongoDB (running on localhost:27017 by default)
- Webcam or video files for input

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download OpenVINO models**:
   - `person-detection-retail-0013.xml` and `.bin`
   - `person-reidentification-retail-0287.xml` and `.bin`
   
   Place them in the `models/` directory.

4. **Set up MongoDB**:
   - Install and start MongoDB
   - Create database `product_analytics`
   - Create collection `zones` with product zone definitions

5. **Add video files** (optional):
   - Place video files in the `videos/` directory
   - Update `CAM_SOURCES` in `app.py` or `.env` file

## Configuration

### Environment Variables (.env file)

Create a `.env` file in the project root with the following variables:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
DB_NAME=product_analytics

# MongoDB Collection Names
ZONES_COLLECTION=zones
FOOTFALL_COL=product_footfall
DWELL_COL=product_dwell
HOURLY_COL=hourly_stats
DAILY_COL=daily_stats
```

If `.env` file is not present, the app will use default values.

### MongoDB Zones Schema

Add product zones to the `zones` collection with the following schema:

```json
{
  "camera_id": 0,
  "product_id": "P101",
  "name": "Laptop Shelf",
  "price": 69999,
  "coordinates": [100, 50, 250, 180]
}
```

Where:
- `camera_id`: Integer index of the camera (0-based)
- `product_id`: Unique string identifier for the product
- `name`: Display name of the product
- `price`: Product price (optional)
- `coordinates`: Array `[x1, y1, x2, y2]` defining the bounding box

### Camera Sources

Update `CAM_SOURCES` in `app.py` to point to your video files or camera indices:

```python
CAM_SOURCES = ["videos/video7.mp4"]  # For video files
# or
CAM_SOURCES = [0, 1]  # For webcam indices
```

## Running the Application

1. **Start MongoDB** (if not already running):
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
# or
mongod
```

2. **Run the Flask app**:
```bash
python app.py
```

3. **Access the dashboard**:
   - Open browser: `http://localhost:5000`
   - Live video feed: `http://localhost:5000/video_feed`
   - API endpoint: `http://localhost:5000/products/live`

## Project Structure

```
.
├── app.py                  # Main Flask application
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── models/                # OpenVINO model files
│   ├── person-detection-retail-0013.xml
│   ├── person-detection-retail-0013.bin
│   ├── person-reidentification-retail-0287.xml
│   └── person-reidentification-retail-0287.bin
└── videos/                # Input video files
    └── video7.mp4
```

## API Endpoints

### `GET /`
Main dashboard page with live video feed and product analytics.

### `GET /video_feed`
Live video stream (multipart/x-mixed-replace) with:
- Person bounding boxes with IDs
- Product zones highlighted
- Heatmap overlay
- Person path tracing

### `GET /products/live`
JSON API returning live product analytics:

```json
{
  "products": {
    "P101": {
      "name": "Laptop Shelf",
      "footfall": 15,
      "dwell_count": 8,
      "avg_dwell": 12.5
    }
  },
  "total_unique": 10,
  "metrics": {}
}
```

## Features Details

### Multi-Camera ReID
- Global person registry maintains unique IDs across all cameras
- Cosine similarity threshold (default: 0.62) for matching persons
- Momentum-based embedding updates for tracking consistency

### Product Zone Analytics
- Footfall: Counts unique person visits per product
- Dwell Time: Tracks how long persons spend in product zones
- Real-time updates as persons enter/exit zones

### Heatmap System
- Gaussian kernel accumulation at person centroids
- Automatic decay (default: 97% per 2 seconds)
- Color-coded overlay on video feed

### Background Workers
- **Hourly Worker**: Saves snapshot of product analytics every hour
- **Daily Worker**: Saves daily summary and resets statistics at midnight

## Troubleshooting

### "ModuleNotFoundError" for any package
Install all dependencies:
```bash
pip install -r requirements.txt
```

### "Cannot connect to MongoDB"
- Ensure MongoDB is running: `mongod` or check service status
- Verify `MONGO_URI` in `.env` file is correct
- Check MongoDB connection: `mongosh` or `mongo`

### "Model file not found"
- Download required OpenVINO models
- Place `.xml` and `.bin` files in `models/` directory
- Verify file paths in `app.py` or `.env`

### "No video frames"
- Check video file path in `CAM_SOURCES`
- Verify video file exists and is readable
- For webcams, check camera index (usually 0 or 1)

### Video keeps looping
The app is configured to play video once. When the video ends, it holds on the last frame.

## Configuration Parameters

Key parameters in `app.py`:

- `REID_SIM_THRESHOLD`: Cosine similarity threshold for person matching (0.62)
- `REID_UPDATE_MOMENTUM`: Momentum for embedding updates (0.6)
- `DETECTION_CONFIDENCE_THRESHOLD`: Minimum confidence for detections (0.55)
- `HEATMAP_DECAY`: Heatmap decay rate per update (0.97)
- `HEATMAP_KERNEL_SIZE`: Gaussian kernel size for heatmap (121)
- `PATH_MAX_LEN`: Maximum path points to store (40)
- `FRAME_THROTTLE`: Frame processing delay in seconds (0.03)

## License

This project is provided as-is for educational and development purposes.

## Notes

- The app requires OpenVINO models to be in the correct format (.xml and .bin files)
- Video files are played once and then hold on the last frame
- Product zones must be defined in MongoDB before analytics will work
- The system is optimized for CPU inference; GPU can be configured via `OPENVINO_DEVICE` in `.env`

The system tracks:
Green dots: People who just entered (< 2 minutes)
Yellow dots: People who stayed 2+ minutes
Orange dots: People who stayed 5+ minutes
Red dots: People who stayed 10+ minutes