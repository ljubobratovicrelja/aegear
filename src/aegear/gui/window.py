import sys
import json
import threading
import time

# Third-party imports
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
from tkinter import ttk

# Internal modules
from aegear.calibration import SceneCalibration
from aegear.trajectory import trajectoryLength, smoothTrajectory, drawTrajectory
from aegear.tracker import FishTracker
from aegear.gui.tracking_bar import TrackingBar
from aegear.utils import resource_path
from aegear.video import VideoClip

# Constants
DEFAULT_CALIBRATION_FILE = resource_path("data/calibration.xml")
HEATMAP_MODEL_PATH = resource_path("data/models/model_efficient_unet_2025-04-04.pth")
SIAMESE_MODEL_PATH = resource_path("data/models/model_siamese_2025-04-20.pth")

class AegearMainWindow(tk.Tk):
    """
    Main application window for Aegear.
    Handles video loading, calibration, tracking, and GUI updates.
    """

    ONE_LINE_HEIGHT = 16  # Will be updated based on the listbox font height.

    def __init__(self):
        super(AegearMainWindow, self).__init__()

        # Hide window during initialization.
        self.withdraw()

        # Set window title.
        self.title("Aegear")

        # Set custom window icon
        icon_path = resource_path("media/icon.ico")
        self.iconbitmap(icon_path)

        # Initialize internal state.
        self._current_frame = None
        self._display_image = None
        self._image_width = 0
        self._playing = False
        self._calibrated = False
        self._calibration_running = False
        self._pixel_to_cm_ratio = 1.0
        self._first_frame_position = None
        self._fish_tracking = {}
        self._trajectory_smooth_size = 9
        self._trajectory_frame_skip = 7
        self._screen_points = []  # Screen points used for calibration.

        # Boolean variable to control drawing the trajectory.
        self._draw_trajectory = tk.BooleanVar(value=True)

        # Initialize the scene calibration utility.
        self._scene_calibration = SceneCalibration(DEFAULT_CALIBRATION_FILE)

        # Ask for initial video file.
        self.dialog_window = tk.Toplevel(self)
        self.dialog_window.withdraw()
        self.dialog_window.title("Load Video")
        initial_video = filedialog.askopenfilename(parent=self.dialog_window)

        # Initialize the fish tracker.
        self._tracker = FishTracker(HEATMAP_MODEL_PATH, SIAMESE_MODEL_PATH, tracking_threshold=0.8, detection_threshold=0.85, debug=False)

        if initial_video == "":
            # No video selected; show error and exit.
            messagebox.showerror("Error", "No video selected.")
            self.destroy()
            sys.exit(1)

        # Load video clip.
        self.clip = VideoClip(initial_video)
        self._num_frames = self.clip.duration * self.clip.fps

        # Set up frames: center for video, bottom for slider/status, right for controls.
        self.center_frame = tk.Frame(self)
        self.center_frame.pack(side=tk.TOP)

        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side=tk.BOTTOM)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT)

        # Create image label to display video frames.
        self.image_label = tk.Label(self.center_frame, cursor="cross")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Create listbox with scrollbar to display tracked frames.
        self.scrollbar = tk.Scrollbar(self.center_frame, orient=tk.VERTICAL)
        self.scrollbar.grid(row=0, column=2, sticky="nsew")

        self.tracking_listbox = tk.Listbox(self.center_frame, width=30, height=10, yscrollcommand=self.scrollbar.set)
        self.tracking_listbox.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights.
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Bind events to the listbox.
        self.tracking_listbox.bind('<<ListboxSelect>>', self._listbox_item_selected)
        self.tracking_listbox.bind('<Delete>', self._listbox_item_deleted)
        self.scrollbar.config(command=self.tracking_listbox.yview)

        # Load the first video frame.
        self._load_first_frame()

        # Create slider and tracking bar.
        self.slider = tk.Scale(self.bottom_frame, from_=0, to=self._num_frames,
                                 length=self._image_width, orient=tk.HORIZONTAL,
                                 command=self.slider_value_changed)
        self.slider.pack()

        self.track_bar = TrackingBar(self.bottom_frame, self._num_frames,
                                     width=self._image_width, height=10)
        self.track_bar.pack()

        # Label to display current video time.
        self.label = tk.Label(self.bottom_frame, text="00:00:00")
        self.label.pack()

        # Create buttons for calibration and tracking control.
        self.calibration_button = tk.Button(self.right_frame, text="Calibrate",
                                            command=self._calibrate, fg="red")
        self.calibration_button.pack(side=tk.LEFT)

        self.set_track_start_button = tk.Button(self.right_frame, text="Set Track Start",
                                                command=self._set_track_start, state=tk.NORMAL)
        self.set_track_start_button.pack(side=tk.LEFT)

        self.set_track_end_button = tk.Button(self.right_frame, text="Set Track End",
                                              command=self._set_track_end, state=tk.NORMAL)
        self.set_track_end_button.pack(side=tk.LEFT)

        self.run_tracking_button = tk.Button(self.right_frame, text="Run Tracking",
                                             command=self._run_tracking, state=tk.NORMAL)
        self.run_tracking_button.pack(side=tk.LEFT)

        self.reset_tracking_button = tk.Button(self.right_frame, text="Reset Tracking",
                                               command=self._reset_tracking)
        self.reset_tracking_button.pack(side=tk.LEFT)

        # Scales for thresholds and tracking parameters.
        self.tracking_threshold_scale = tk.Scale(self.right_frame, from_=0, to=100,
                                                 orient=tk.HORIZONTAL, label="Tracking Threshold",
                                                 command=self._tracking_threshold_changed)
        self.tracking_threshold_scale.set(int(self._tracker.siamese_threshold * 100.0))
        self.tracking_threshold_scale.pack(side=tk.LEFT)

        self.detection_threshold_scale = tk.Scale(self.right_frame, from_=0, to=100,
                                                  orient=tk.HORIZONTAL, label="Detection Threshold",
                                                  command=self._detection_threshold_changed)
        self.detection_threshold_scale.set(int(self._tracker.heatmap_threshold * 100.0))
        self.detection_threshold_scale.pack(side=tk.LEFT)

        self.smooth_trajectory_scale = tk.Scale(self.right_frame, from_=1, to=100,
                                                orient=tk.HORIZONTAL, label="Trajectory Smoothing",
                                                command=self._trajectory_smooth_size_changed)
        self.smooth_trajectory_scale.set(self._trajectory_smooth_size)
        self.smooth_trajectory_scale.pack(side=tk.LEFT)

        self.tracking_frame_scale = tk.Scale(self.right_frame, from_=1, to=100,
                                             orient=tk.HORIZONTAL, label="Tracking Frame Skip",
                                             command=self._trajectory_frame_skip_changed)
        self.tracking_frame_scale.set(self._trajectory_frame_skip)
        self.tracking_frame_scale.pack(side=tk.LEFT)

        self.draw_trajectory_checkbox = tk.Checkbutton(self.right_frame, text="Draw Trajectory",
                                                       variable=self._draw_trajectory)
        self.draw_trajectory_checkbox.pack(side=tk.LEFT)

        # Navigation buttons.
        self.prev_frame_button = tk.Button(self.right_frame, text=u"\u23EE",
                                           command=self.previous_frame)
        self.prev_frame_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.right_frame, text=u"\u23F8",
                                      command=self._pause_tracking)
        self.pause_button.pack(side=tk.RIGHT)

        self.play_button = tk.Button(self.right_frame, text=u"\u25B6",
                                     command=self._start_tracking)
        self.play_button.pack(side=tk.RIGHT)

        self.next_frame_button = tk.Button(self.right_frame, text=u"\u23ED",
                                           command=self.next_frame)
        self.next_frame_button.pack(side=tk.RIGHT)

        # Status bars.
        self.status_bar = tk.Label(self.bottom_frame, text="Not Calibrated", fg="red",
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.distance_status_bar = tk.Label(self.bottom_frame, text="Distance: 0.0 cm", fg="blue",
                                            bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.distance_status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Final GUI update and menu creation.
        self._setup_keypress_events()
        self._setup_image_mouse_events()
        self.update_gui()
        self._create_menu()

        # Show window after setup.
        self.deiconify()
    
    def _setup_keypress_events(self):
        """Set keypress events for navigation."""
        self.bind("<Left>", lambda _: self._previous_tracked_frame())
        self.bind("<Right>", lambda _: self._next_tracked_frame())
        self.bind("<space>", lambda _: self._start_tracking())
        self.bind("<Escape>", lambda _: self._pause_tracking())
        self.bind("<Delete>", lambda _: self._delete_current_tracked_frame())
    
    def _setup_image_mouse_events(self):
        """Set mouse events for image interactions."""
        self.image_label.bind("<Button-1>", lambda event: self._add_tracking_point(event))
        self.image_label.bind("<Button-3>", lambda event: self._remove_current_frame_track(event))
        self.image_label.bind("<MouseWheel>", lambda event: self._seek_frames(event))
    
    def _delete_current_tracked_frame(self):
        """Delete the currently tracked frame from the list."""
        current_frame = self._get_current_frame_number()
        if current_frame in self._fish_tracking:
            del self._fish_tracking[current_frame]
        self._rebuild_tracking_listbox()
    
    def _rebuild_tracking_listbox(self):
        """Rebuild the tracking listbox from the current fish tracking data."""
        self.tracking_listbox.delete(0, tk.END)
        for frame_id, (coordinates, confidence) in sorted(self._fish_tracking.items()):
            self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))
        self.update_gui()
    
    def _add_tracking_point(self, event):
        print("Adding tracking point at ({}, {})".format(event.x, event.y))
        current_frame = self._get_current_frame_number()
        self._insert_tracking_point(current_frame, (event.x, event.y), 1.0)
    
    def _next_tracked_frame(self):
        """Move to the next tracked frame."""
        if len(self._fish_tracking) < 1:
            return

        current_frame = self._get_current_frame_number()
        next_frame = current_frame + 1

        while next_frame not in self._fish_tracking:
            if next_frame >= self._num_frames:
                return
            next_frame += 1

        self._play_frame(next_frame)

    def _previous_tracked_frame(self):
        """Move to the previous tracked frame."""
        if len(self._fish_tracking) < 1:
            return

        current_frame = self._get_current_frame_number()
        previous_frame = current_frame - 1

        while previous_frame not in self._fish_tracking:
            previous_frame -= 1

            if previous_frame < 0:
                return

        self._play_frame(previous_frame)
    
    def _remove_current_frame_track(self, event):
        current_frame = self._get_current_frame_number()
        if current_frame in self._fish_tracking:
            del self._fish_tracking[current_frame]
            self.tracking_listbox.delete(tk.END)
        self.update_gui()
    
    def _seek_frames(self, event):
        """Move to the tracked frame based on mouse wheel scroll."""
        if event.delta > 0:
            self.next_frame()
        else:
            self.previous_frame()
    
    def _insert_tracking_point(self, frame_id, coordinates, confidence):
        """Insert a tracking point into the listbox."""
        self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))
        self._fish_tracking[frame_id] = (coordinates, confidence)
        self.track_bar.mark_processed(frame_id)
        self.update_gui()

    def _create_menu(self):
        """Set up the application menu bar."""
        self.menu_bar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load Video", command=self._load_video)
        self.file_menu.add_command(label="Load Tracking", command=self._load_tracking)
        self.file_menu.add_command(label="Save Tracking", command=self._save_tracking)
        self.file_menu.add_command(label="Save Trajectory", command=self._save_trajectory)
        self.file_menu.add_command(label="Load Scene Reference", command=self._load_scene_reference)
        self.file_menu.add_command(label="Exit", command=self.destroy)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self._about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menu_bar)

    def _listbox_item_selected(self, event):
        """
        Callback when an item in the tracking listbox is selected.
        Updates the current frame to the selected frame.
        """
        sel = self.tracking_listbox.curselection()
        if not sel:
            return

        item_selected = sel[0]
        # Extract frame id from the listbox item (format: "frame_id: score")
        frame_id = int(self.tracking_listbox.get(item_selected).split(":")[0])
        self._set_frame(frame_id)
        self.update_gui()

    def _listbox_item_deleted(self, event):
        """
        Callback when an item in the tracking listbox is deleted.
        Removes the corresponding processed frame marker.
        """
        item_selected = self.tracking_listbox.curselection()[0]
        frame_id = int(self.tracking_listbox.get(item_selected).split(":")[0])
        self.track_bar.mark_not_processed(frame_id)
        del self._fish_tracking[frame_id]
        self.tracking_listbox.delete(item_selected)
        self.update_gui()

    def _tracking_threshold_changed(self, value):
        """Update the tracking threshold for the fish tracker."""
        self._tracker.siamese_threshold = float(value) / 100.0

    def _detection_threshold_changed(self, value):
        """Update the detection threshold for the fish tracker."""
        self._tracker.heatmap_threshold = float(value) / 100.0

    def _trajectory_smooth_size_changed(self, value):
        """
        Update the smoothing size for trajectory.
        Enforces an odd value for the smoothing kernel.
        """
        v = int(value)
        if v % 2 == 0:  # Ensure the smoothing size is odd.
            v += 1
        self._trajectory_smooth_size = v
        self.smooth_trajectory_scale.set(v)
        self.update_gui()

    def _trajectory_frame_skip_changed(self, value):
        """Update the frame skip value used during tracking."""
        self._trajectory_frame_skip = int(value)

    def _run_tracking(self):
        """
        Run the fish tracking process over a range of frames.
        Displays progress and estimates remaining time.
        """
        if self._tracker is None:
            messagebox.showerror("Error", "Tracking model not initialized.")
            return

        if self.track_bar.processing_start is None or self.track_bar.processing_end is None:
            messagebox.showerror("Error", "Please set the processing start and end frames.")
            return

        # Create a modal task window for tracking progress.
        task_window = tk.Toplevel(self)
        task_window.title("Tracking")
        task_window.geometry("300x120")

        progress_label = tk.Label(task_window, text="Progress: 0%")
        progress_label.pack()

        progress = ttk.Progressbar(task_window, length=200)
        progress.pack(pady=20)

        cancel_button = tk.Button(task_window, text="Cancel", command=task_window.destroy)
        cancel_button.pack()

        progress['value'] = 0
        track_start_frame = self.track_bar.processing_start
        track_end_frame = self.track_bar.processing_end

        # Calculate progress increment per frame processed.
        progress_increment = 100.0 / ((track_end_frame - track_start_frame) / self._trajectory_frame_skip)
        progress_value = 0.0

        # Initialize background subtractor for motion detection.
        bck_substractor = cv2.createBackgroundSubtractorKNN(history=100)

        # Warm up background subtractor using preceding frames.
        for frame_id in range(max(track_start_frame - 10, 0), track_start_frame):
            frame_image = self._read_frame(frame_id)
            if frame_image is None:
                continue
            gframe_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            gframe_image = cv2.medianBlur(gframe_image, 3)
            bck_substractor.apply(gframe_image)

        start_time = time.time()

        for frame_id in range(track_start_frame, track_end_frame, self._trajectory_frame_skip):
            # Skip first and last frames where tracking context is insufficient.
            if frame_id == 0 or frame_id == self._num_frames:
                continue

            progress_value += progress_increment
            progress['value'] = progress_value

            # Exact time in seconds of the current frame for tracker history and trajectory prediciton mechanism.
            frame_t = float(frame_id) / float(self.clip.fps)

            elapsed_time = time.time() - start_time
            estimated_time = elapsed_time / progress_value * 100.0 - elapsed_time
            estimated_time_str = "{:02d}:{:02d}:{:02d}".format(
                int(estimated_time // 3600),
                int((estimated_time // 60) % 60),
                int(estimated_time % 60)
            )
            progress_label['text'] = "Progress: {}%, estimated time: {}".format(int(progress_value), estimated_time_str)

            # Read and process the current frame.
            frame_image = self._read_frame(frame_id)
            gframe_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
            gframe_image = cv2.medianBlur(gframe_image, 3)
            background_mask = bck_substractor.apply(gframe_image)

            # Apply morphological operations to reduce noise.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            background_mask = cv2.erode(background_mask, kernel, iterations=1)
            background_mask = cv2.dilate(background_mask, kernel, iterations=1)
            _, background_mask = cv2.threshold(background_mask, 127, 255, cv2.THRESH_BINARY)

            draw_image = frame_image.copy()
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

            result = self._tracker.track(frame_image, frame_t, mask=background_mask)
            if result is not None:
                (coordinates, confidence) = result.centroid, result.confidence
                self.track_bar.mark_processed(frame_id)
                cv2.circle(draw_image, coordinates, 5, (255, 0, 0), -1)
                self._insert_tracking_point(frame_id, coordinates, confidence)

            self._display_image = draw_image.copy()
            self.update_image(self._display_image)
            self.update()

        task_window.transient(self)
        task_window.grab_set()
        task_window.destroy()

        try:
            self.wait_window(task_window)
        except Exception:
            pass

        self._play_frame(track_start_frame)

    def _set_track_start(self):
        """Set the current slider position as the start of processing."""
        self.track_bar.mark_processing_start(self.slider.get())

    def _set_track_end(self):
        """Set the current slider position as the end of processing."""
        self.track_bar.mark_processing_end(self.slider.get())

    def _reset_tracking(self):
        """Reset all tracking markers and clear the tracking list."""
        self.track_bar.clear()
        self._fish_tracking = {}
        self.tracking_listbox.delete(0, tk.END)
        self.track_bar.unmark_processing_start()
        self.track_bar.unmark_processing_end()
        self.update_gui()

    def _about(self):
        """Display information about the application."""
        messagebox.showinfo("About", "Aegear\n\nAuthor: Relja Ljubobratovic\nEmail: ljubobratovic.relja@gmail.com")

    def _load_tracking(self):
        filename = filedialog.askopenfilename(defaultextension=".json",
                                              filetypes=[("JSON files", "*.json")])
        if filename == "":
            messagebox.showerror("Error", "No file selected.")
            return
        
        try:
            with open(filename, 'r') as f:
                file_dict = json.load(f)
                if "video" not in file_dict or "tracking" not in file_dict:
                    raise ValueError("Invalid file format.")           

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tracking data: {e}")
            return
        
        self._fish_tracking = {}
        for item in file_dict["tracking"]:
            frame_id = item["frame_id"]
            coordinates = tuple(item["coordinates"])
            confidence = item["confidence"]
            self._fish_tracking[frame_id] = (coordinates, confidence)
            self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))

    def _save_tracking(self):
        if self._fish_tracking is None or len(self._fish_tracking) == 0:
            messagebox.showerror("Error", "No tracking data available.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json")])
        if filename == "":
            messagebox.showerror("Error", "No file selected.")
            return
        
        file_dict = {
            "video": self.clip.filename,
            "tracking": []
        }

        for frame_id, (coordinates, confidence) in self._fish_tracking.items():
            file_dict["tracking"].append({
                "frame_id": frame_id,
                "coordinates": coordinates,
                "confidence": confidence
            })
        
        with open(filename, 'w') as f:
            json.dump(file_dict, f, indent=4)

        self.status_bar['text'] = "Tracking data saved to {}".format(filename)
        self.status_bar['fg'] = "green"

    def _save_trajectory(self):
        if self._fish_tracking is None or len(self._fish_tracking) == 0:
            messagebox.showerror("Error", "No tracking data available.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                  filetypes=[("JSON files", "*.json")])
        if filename == "":
            messagebox.showerror("Error", "No file selected.")
            return
        
        file_dict = {
            "video": self.clip.filename,
            "trajectory": []
        }

        trajectory = []
        for (coordinates, _) in self._fish_tracking.values():
            trajectory.append(coordinates)
        
        trajectory = smoothTrajectory(trajectory, self._trajectory_smooth_size)
        start_frame = next(iter(self._fish_tracking))

        if start_frame is None:
            raise ValueError("Failed finding starting frame for trajectory.")

        for (frame_id, coordinate) in enumerate(trajectory):
            frame_id += start_frame
            file_dict["trajectory"].append({
                "frame_id": frame_id,
                "coordinates": coordinate
            })
        
        with open(filename, 'w') as f:
            json.dump(file_dict, f, indent=4)

        self.status_bar['text'] = "Trajectory saved to {}".format(filename)
        self.status_bar['fg'] = "green"

    def _load_scene_reference(self):
        # TODO: Implement loading of a scene reference for calibration.
        pass

    def _load_video(self):
        """Load a new video file and reinitialize the video processing."""
        filename = filedialog.askopenfilename()
        if filename == "":
            messagebox.showerror("Error", "No video selected.")
            return

        try:
            self.clip = VideoClip(filename)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            self.clip = None

        if self.clip is not None:
            self._calibrated = False
            self._calibration_running = False
            self._playing = False
            self._current_frame = None
            self._screen_points = []
            self._pixel_to_cm_ratio = 1.0

            slider_length = self.clip.duration * self.clip.fps
            self.slider.destroy()
            self.slider = tk.Scale(self.bottom_frame, from_=0, to=slider_length,
                                   length=self._image_width, orient=tk.HORIZONTAL,
                                   command=self.slider_value_changed)
            self.slider.pack()
            self.slider.set(0)
            self._load_first_frame()

    def _calibrate(self):
        """
        Start or cancel calibration.
        Left-clicks on the image will select calibration points until four are set.
        """
        if self._calibrated:
            if not messagebox.askokcancel("Calibration Reset", "Are you sure you want to reset calibration?"):
                return
        elif self._calibration_running:
            if messagebox.askokcancel("Calibration Cancel", "Are you sure you want to cancel calibration?"):
                self.status_bar['text'] = "Calibration cancelled."
                self.status_bar['fg'] = "red"
                self._screen_points = []
                self.calibration_button['text'] = "Calibrate"
                self.calibration_button['fg'] = "red"
                self.image_label.unbind("<Button-1>")
                self._calibration_running = False
                self._calibrated = False
                self.update_gui()
                return

        self._calibrated = False
        self._calibration_running = True
        self._screen_points = []
        self.update_gui()
        self.status_bar['text'] = "Calibration started - left click to select corner points of the scene."
        self.status_bar['fg'] = "orange"
        self.image_label.bind("<Button-1>", self._calibration_click)
        self.calibration_button['text'] = "Cancel Calibration"
        self.calibration_button['fg'] = "purple"

    def _do_calibration(self):
        """
        Perform the calibration once four screen points are collected.
        Updates the pixel-to-cm ratio and GUI state accordingly.
        """
        try:
            self._pixel_to_cm_ratio = self._scene_calibration.calibrate(self._screen_points)
            self._calibrated = True
            self._calibration_running = False
            self.status_bar['text'] = "Calibration complete: pixel to cm ratio is {}".format(self._pixel_to_cm_ratio)
            self.status_bar['fg'] = "green"
            self.calibration_button['text'] = "Reset Calibration"
            self.calibration_button['fg'] = "green"
            self.set_track_start_button["state"] = tk.NORMAL
            self.set_track_end_button["state"] = tk.NORMAL
            self.run_tracking_button["state"] = tk.NORMAL
        except Exception as e:
            self._calibrated = False
            self._calibration_running = False
            self.status_bar['text'] = "Calibration failed - internal error: {}".format(e)
            self.status_bar['fg'] = "red"
            self.calibration_button['text'] = "Calibrate"
            self.calibration_button['fg'] = "red"

        self._screen_points = []
        self.image_label.unbind("<Button-1>")
        self._reload_frame()
        self.update_gui()

    def _calibration_click(self, event):
        """
        Handle a click during calibration.
        Record the clicked point and update GUI.
        Once four points are recorded, perform calibration.
        """
        self._screen_points.append((event.x, event.y))
        self.update_gui()
        if len(self._screen_points) == 4:
            self._do_calibration()
            return
        self.status_bar['text'] = "Calibration point {} selected.".format(len(self._screen_points))

    def _get_current_frame_number(self):
        """Return the current frame number from the slider."""
        return self.slider.get()

    def _start_tracking(self):
        """
        Start playing frames.
        If already playing, this call stops the playback.
        """
        if self._playing:
            self._playing = False
            return

        self._playing = True
        threading.Thread(target=self._read_frames).start()

    def _read_frame(self, frame_number):
        """
        Retrieve a frame from the video clip.
        If calibrated, the frame is rectified.
        """
        frame = self.clip.get_frame(float(frame_number) / float(self.clip.fps))
        if self._calibrated:
            frame = self._scene_calibration.rectify_image(frame)
        return frame
    
    def _read_current_frame(self):
        """Return the current frame based on the slider."""
        return self._read_frame(self._get_current_frame_number())

    def _read_previous_frame(self):
        """Return the previous frame (one frame back)."""
        return self._read_frame(self._get_current_frame_number() - 1)

    def _pause_tracking(self):
        """Pause frame playback."""
        self._playing = False

    def _reload_frame(self):
        """Reload the current frame and update display image."""
        self._current_frame = self._read_current_frame()
        self._display_image = self._current_frame.copy()

    def _set_frame(self, frame):
        """Set the slider to a specific frame."""
        self.slider.set(frame)

    def previous_frame(self):
        """Move to the previous frame."""
        self._play_frame(self.slider.get() - 1)

    def next_frame(self):
        """Move to the next frame."""
        self._play_frame(self.slider.get() + 1)

    def _play_frame(self, frame):
        """
        Set a specific frame and update the GUI accordingly.
        """
        self._set_frame(frame)
        self._reload_frame()
        self.update_gui()

    def _read_frames(self):
        """
        Read and play frames in a loop while playback is active.
        Uses a short sleep to synchronize with the video FPS.
        """
        while self._playing:
            time.sleep(0.5 / self.clip.fps)
            self.after(1, self.next_frame)

    def update_gui(self):
        """
        Update the image display and the time label based on the current frame.
        """
        self.update_image(self._display_image)
        self.label['text'] = self._frame_to_time(float(self._get_current_frame_number()), self.clip.fps)

    def _draw_motion(self, motion):
        """
        Overlay the motion detection result on the display image.
        """
        motion_overlay = cv2.resize(motion, (self._display_image.shape[1], self._display_image.shape[0]))
        motion_overlay = cv2.cvtColor(motion_overlay, cv2.COLOR_GRAY2BGR)
        motion_overlay = (motion_overlay * 255).astype(np.uint8)  # Scale mask to 255.
        self._display_image = cv2.addWeighted(self._display_image, 0.5, motion_overlay, 0.5, 0)

    def update_image(self, image):
        """
        Update the image displayed in the GUI.
        Draws calibration points, tracked positions, and trajectory.
        """
        if image is None:
            return

        image = image.copy()

        # Draw calibration points.
        for point in self._screen_points:
            image = cv2.circle(image, point, 5, (0, 0, 255), -1)

        # Draw current tracked point if available.
        frame_id = self._get_current_frame_number()
        if frame_id in self._fish_tracking:
            ((x, y), _) = self._fish_tracking[frame_id]
            image = cv2.circle(image, (x, y), 12, (0, 255, 0))

        # Draw trajectory if enabled.
        if self._draw_trajectory.get() and len(self._fish_tracking) > 1:
            trajectory = []
            tracked_frames = sorted(list(self._fish_tracking.keys()))
            for frame_id in tracked_frames:
                if frame_id >= self._get_current_frame_number():
                    break
                centroid, _ = self._fish_tracking[frame_id]
                trajectory.append(centroid)

            s_trajectory = smoothTrajectory(trajectory, self._trajectory_smooth_size)
            travelDistance = trajectoryLength(s_trajectory) * self._pixel_to_cm_ratio
            self.distance_status_bar['text'] = "Distance: {} cm".format(travelDistance)
            image = drawTrajectory(image, s_trajectory)

        # Convert image from OpenCV format to PIL, then to Tkinter-compatible image.
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=image)
        self.image_label.image = image  # Prevent garbage collection.

    def slider_value_changed(self, value):
        """
        Callback for slider value changes.
        Reloads and displays the corresponding frame.
        """
        self._current_frame = self._read_frame(int(value))
        self._display_image = self._current_frame.copy()
        self.update_gui()

    def _load_first_frame(self):
        """
        Load the first frame from the video.
        Also adjusts the tracking listbox height based on the frame size.
        """
        self._current_frame = self._read_frame(0)
        self._display_image = self._current_frame.copy()
        assert self._current_frame is not None, "Failed to load first frame."
        self._image_width = self._current_frame.shape[1]

        # Use a temporary Text widget to determine one-line height.
        temp_text = tk.Text(self.tracking_listbox, height=1, font=("TkDefaultFont"))
        temp_text.pack()
        self.tracking_listbox.update()
        AegearMainWindow.ONE_LINE_HEIGHT = temp_text.winfo_reqheight() + 4  # Add margin.
        temp_text.destroy()

        self.tracking_listbox.config(height=int(self._current_frame.shape[0] / AegearMainWindow.ONE_LINE_HEIGHT))
        self.tracking_listbox.update()

    def _frame_to_time(self, frame, fps):
        """
        Convert a frame number to a formatted time string (hh:mm:ss).
        """
        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
