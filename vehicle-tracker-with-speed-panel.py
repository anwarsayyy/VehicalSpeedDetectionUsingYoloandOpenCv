import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from PIL import Image, ImageTk

class VehicleTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Speed Tracking System")
        
        # Initialize model and tracking variables
        self.model = YOLO('yolov8s.pt')
        self.cap = None
        self.is_running = False
        self.video_path = None
        
        # Vehicle tracking data
        self.vehicle_data = defaultdict(lambda: {
            'type': None,
            'speeds': [],
            'positions': [],
            'timestamps': [],
            'first_seen': None,
            'last_seen': None,
            'active': True  # New field to track active vehicles
        })
        
        # Speed estimation parameters
        self.reference_distance = 20
        self.reference_pixels = 200
        self.meters_per_pixel = self.reference_distance / self.reference_pixels
        
        # Define reference lines
        self.ref_line1_y = 300
        self.ref_line2_y = 400
        
        # Vehicle classes
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
        }
        
        # Speed calculation parameters
        self.speed_update_interval = 1.0
        self.last_speed_update = time.time()
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for video and controls
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for speed display
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display canvas (in left panel)
        self.canvas = tk.Canvas(self.left_panel, width=800, height=500)
        self.canvas.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Controls frame
        controls_frame = ttk.Frame(self.left_panel)
        controls_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Calibration frame
        calibration_frame = ttk.LabelFrame(self.left_panel, text="Calibration", padding="5")
        calibration_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Label(calibration_frame, text="Reference Distance (m):").grid(row=0, column=0, padx=5)
        self.ref_distance_var = tk.StringVar(value="20")
        ttk.Entry(calibration_frame, textvariable=self.ref_distance_var).grid(row=0, column=1, padx=5)
        
        # Buttons
        ttk.Button(controls_frame, text="Select Video", command=self.select_video).grid(row=0, column=0, padx=5)
        ttk.Button(controls_frame, text="Start", command=self.start_tracking).grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="Stop", command=self.stop_tracking).grid(row=0, column=2, padx=5)
        ttk.Button(controls_frame, text="Calibrate", command=self.calibrate).grid(row=0, column=3, padx=5)
        
        # Statistics frame
        self.stats_frame = ttk.LabelFrame(self.left_panel, text="Vehicle Statistics", padding="5")
        self.stats_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # Treeview for vehicle statistics
        self.tree = ttk.Treeview(self.stats_frame, columns=('Type', 'Count', 'Avg Speed', 'Max Speed'), show='headings')
        self.tree.heading('Type', text='Vehicle Type')
        self.tree.heading('Count', text='Count')
        self.tree.heading('Avg Speed', text='Avg Speed (km/h)')
        self.tree.heading('Max Speed', text='Max Speed (km/h)')
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Setup speed panel (right side)
        self.setup_speed_panel()

    def setup_speed_panel(self):
        # Create speed panel frame with a title
        speed_panel_frame = ttk.LabelFrame(self.right_panel, text="Live Vehicle Speeds", padding="10")
        speed_panel_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Create Treeview for live speeds
        self.speed_tree = ttk.Treeview(
            speed_panel_frame,
            columns=('ID', 'Type', 'Current Speed'),
            show='headings',
            height=20
        )
        
        # Configure columns
        self.speed_tree.heading('ID', text='Vehicle ID')
        self.speed_tree.heading('Type', text='Type')
        self.speed_tree.heading('Current Speed', text='Speed (km/h)')
        
        # Set column widths
        self.speed_tree.column('ID', width=80)
        self.speed_tree.column('Type', width=100)
        self.speed_tree.column('Current Speed', width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(speed_panel_frame, orient=tk.VERTICAL, command=self.speed_tree.yview)
        self.speed_tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid the Treeview and scrollbar
        self.speed_tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def update_speed_panel(self):
        # Clear existing entries
        for item in self.speed_tree.get_children():
            self.speed_tree.delete(item)
        
        current_time = time.time()
        
        # Add active vehicles to the speed panel
        for vehicle_id, data in self.vehicle_data.items():
            # Check if vehicle is still active (seen in last 2 seconds)
            if current_time - data['last_seen'] < 2.0 and data['speeds']:
                # Get the latest speed
                current_speed = data['speeds'][-1] if data['speeds'] else 0
                
                # Add to speed tree
                self.speed_tree.insert('', 'end', values=(
                    f"#{vehicle_id}",
                    data['type'],
                    f"{current_speed:.1f}"
                ))

    def calculate_speed(self, vehicle_id, current_position, current_time):
        vehicle = self.vehicle_data[vehicle_id]
        
        if not vehicle['positions'] or not vehicle['timestamps']:
            vehicle['positions'].append(current_position)
            vehicle['timestamps'].append(current_time)
            return 0

        prev_position = vehicle['positions'][-1]
        prev_time = vehicle['timestamps'][-1]
        time_diff = current_time - prev_time

        if time_diff < 0.1:
            return vehicle['speeds'][-1] if vehicle['speeds'] else 0

        pixel_distance = np.sqrt(
            (current_position[0] - prev_position[0]) ** 2 +
            (current_position[1] - prev_position[1]) ** 2
        )

        distance_meters = pixel_distance * self.meters_per_pixel
        speed = (distance_meters / time_diff) * 3.6

        if vehicle['speeds']:
            speed = (speed + vehicle['speeds'][-1]) / 2

        vehicle['positions'].append(current_position)
        vehicle['timestamps'].append(current_time)
        
        if len(vehicle['positions']) > 10:
            vehicle['positions'].pop(0)
            vehicle['timestamps'].pop(0)

        return min(max(speed, 0), 200)

    def process_frame(self, frame):
        # Draw reference lines
        cv2.line(frame, (0, self.ref_line1_y), (frame.shape[1], self.ref_line1_y), (255, 0, 0), 2)
        cv2.line(frame, (0, self.ref_line2_y), (frame.shape[1], self.ref_line2_y), (255, 0, 0), 2)

        results = self.model.track(frame, persist=True, classes=list(self.vehicle_classes.keys()))
        
        current_time = time.time()
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            classes = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                v_type = self.vehicle_classes[int(cls)]
                
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                current_position = (center_x, center_y)
                
                if track_id not in self.vehicle_data:
                    self.vehicle_data[track_id]['type'] = v_type
                    self.vehicle_data[track_id]['first_seen'] = current_time
                
                self.vehicle_data[track_id]['last_seen'] = current_time
                
                speed = self.calculate_speed(track_id, current_position, current_time)
                if speed > 0:
                    self.vehicle_data[track_id]['speeds'].append(speed)
                
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                if self.vehicle_data[track_id]['speeds']:
                    avg_speed = sum(self.vehicle_data[track_id]['speeds'][-5:]) / min(len(self.vehicle_data[track_id]['speeds']), 5)
                    speed_text = f"{v_type}: {avg_speed:.1f} km/h"
                    cv2.putText(frame, speed_text, (int(box[0]), int(box[1]-10)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                positions = self.vehicle_data[track_id]['positions']
                if len(positions) >= 2:
                    points = np.array(positions, np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], False, color, 2)
        
        return frame

    def update_frame(self):
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (800, 500))  # Adjusted size to make room for speed panel
                frame = self.process_frame(frame)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
                
                self.update_statistics()
                self.update_speed_panel()  # Update the speed panel
                
                self.root.after(10, self.update_frame)
            else:
                self.stop_tracking()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            
    def update_statistics(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        stats = defaultdict(lambda: {'count': 0, 'speeds': []})
        
        for vehicle_info in self.vehicle_data.values():
            if vehicle_info['type'] in self.vehicle_classes.values():
                v_type = vehicle_info['type']
                stats[v_type]['count'] += 1
                stats[v_type]['speeds'].extend(vehicle_info['speeds'])
                
        for v_type, data in stats.items():
            speeds = data['speeds']
            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            max_speed = max(speeds) if speeds else 0
            
            self.tree.insert('', 'end', values=(
                v_type,
                data['count'],
                f"{avg_speed:.1f}",
                f"{max_speed:.1f}"
            ))

    def calibrate(self):
        try:
            self.reference_distance = float(self.ref_distance_var.get())
            self.meters_per_pixel = self.reference_distance / self.reference_pixels
        except ValueError:
            pass
            
    def start_tracking(self):
        if self.cap is not None and not self.is_running:
            self.is_running = True
            self.update_frame()
            
    def stop_tracking(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleTracker(root)
    root.mainloop()
