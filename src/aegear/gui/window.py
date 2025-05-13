import sys
import json
import threading
import time

# Third-party imports
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

# Internal modules
from aegear.calibration import SceneCalibration
from aegear.trajectory import trajectory_length, smooth_trajectory, draw_trajectory
from aegear.tracker import FishTracker

from aegear.gui.tracking_bar import TrackingBar
from aegear.gui.progress_reporter import ProgressReporter
from aegear.gui.video_canvas import VideoCanvas

from aegear.utils import resource_path
from aegear.video import VideoClip

# Constants
DEFAULT_CALIBRATION_FILE = resource_path("data/calibration.xml")
HEATMAP_MODEL_PATH = resource_path("data/models/model_efficient_unet_2025-05-11.pth")
SIAMESE_MODEL_PATH = resource_path("data/models/model_siamese_2025-05-12.pth")

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
        self._playing = False
        self._calibrated = False
        self._calibration_running = False
        self._pixel_to_cm_ratio = 1.0
        self._first_frame_position = None
        self._fish_tracking = {}
        self._trajectory_smooth_size = 5
        self._smooth_trajectory = None
        self._screen_points = []
        self._num_frames = 100
        self._clip = None

        # Boolean variable to control drawing the trajectory.
        self._draw_trajectory = tk.BooleanVar(value=True)

        # Initialize the scene calibration utility.
        self._scene_calibration = SceneCalibration(DEFAULT_CALIBRATION_FILE)

        # Ask for initial video file.
        self.dialog_window = tk.Toplevel(self)
        self.dialog_window.withdraw()
        self.dialog_window.title("Load Video")

        # Initialize the fish tracker.
        self._tracker = FishTracker(HEATMAP_MODEL_PATH,
                                    SIAMESE_MODEL_PATH,
                                    tracking_threshold=0.9,
                                    detection_threshold=0.9,
                                    debug=False,
                                    search_stride=0.5,
                                    tracking_max_skip=7)


        self._setup_main_ui_layout()
        self._load_first_frame()
        self._setup_keypress_events()
        self._setup_image_mouse_events()
        self._create_menu()

        # Final tk touches.
        self.update_idletasks()

        # Set the window size to the minimum required size.
        min_w = self.winfo_reqwidth()
        min_h = self.winfo_reqheight()
        min_h = max(min_h, 600)
        min_w = max(min_w, 900)
        self.minsize(min_w, min_h)

        self.deiconify()
    
    def _setup_main_ui_layout(self):
        """
        Set up the main UI layout with a PanedWindow and frames for different sections."""

        self.main_pane = tk.PanedWindow(self,
                                        orient=tk.HORIZONTAL,
                                        sashrelief=tk.RAISED,
                                        bd=2,
                                        bg='lightgrey',
                                        sashwidth=6)
        self.main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        self.bottom_controls_frame = tk.Frame(self)
        self.bottom_controls_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=5)

        self.left_toolbox_frame = tk.Frame(self.main_pane, bd=2, relief=tk.RAISED)
        self.main_pane.add(self.left_toolbox_frame, minsize=180)

        self.center_video_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.center_video_frame, minsize=400)

        self.right_listbox_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.right_listbox_frame, minsize=150)

        self.main_pane.paneconfig(self.left_toolbox_frame, stretch="never")
        self.main_pane.paneconfig(self.center_video_frame, stretch="always")
        self.main_pane.paneconfig(self.right_listbox_frame, stretch="never")

        def _initialize_sash_positions():
            if self.main_pane.winfo_ismapped() and self.main_pane.winfo_width() > 1:
                total_width = self.main_pane.winfo_width()
                left_width = int(total_width * 0.20)
                center_width = int(total_width * 0.60)
                sash0_pos = left_width
                sash1_pos = left_width + center_width

                try:
                    self.main_pane.sash_place(0, sash0_pos, 0)
                    self.main_pane.sash_place(1, sash1_pos, 0)
                except tk.TclError as e:
                    print(f"TclError setting sash positions: {e}")
            else:
                self.after(50, _initialize_sash_positions)

        self.after_idle(_initialize_sash_positions)

        # Toolbox frame
        toolbox_padding = {'pady': 3, 'padx': 5}

        calib_frame = tk.LabelFrame(self.left_toolbox_frame, text="Calibration")
        calib_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)

        self.calibration_button = tk.Button(calib_frame, text="Calibrate", command=self._calibrate, fg="red")
        self.calibration_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        track_control_frame = tk.LabelFrame(self.left_toolbox_frame, text="Tracking Control")
        track_control_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)

        self.set_track_start_button = tk.Button(track_control_frame, text="Set Track Start", command=self._set_track_start)
        self.set_track_start_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)
        self.set_track_end_button = tk.Button(track_control_frame, text="Set Track End", command=self._set_track_end)
        self.set_track_end_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        self.run_tracking_button = tk.Button(track_control_frame, text="Run Tracking", command=self._run_tracking)
        self.run_tracking_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        self.reset_tracking_button = tk.Button(track_control_frame, text="Reset Tracking", command=self._reset_tracking)
        self.reset_tracking_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        track_params_frame = tk.LabelFrame(self.left_toolbox_frame, text="Parameters")
        track_params_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)

        self.tracking_threshold_scale = tk.Scale(track_params_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Tracking Thresh", command=self._tracking_threshold_changed)
        self.tracking_threshold_scale.set(int(self._tracker.tracking_threshold * 100.0))
        self.tracking_threshold_scale.pack(side=tk.TOP, fill=tk.X, pady=1, padx=5)

        self.detection_threshold_scale = tk.Scale(track_params_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Detection Thresh", command=self._detection_threshold_changed)
        self.detection_threshold_scale.set(int(self._tracker.detection_threshold * 100.0))
        self.detection_threshold_scale.pack(side=tk.TOP, fill=tk.X, pady=1, padx=5)

        self.smooth_trajectory_scale = tk.Scale(track_params_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="Trajectory Smoothing", command=self._trajectory_smooth_size_changed)
        self.smooth_trajectory_scale.set(self._trajectory_smooth_size)
        self.smooth_trajectory_scale.pack(side=tk.TOP, fill=tk.X, pady=1, padx=5)

        self.tracking_frame_scale = tk.Scale(track_params_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="Frame Skip", command=self._trajectory_frame_skip_changed)
        self.tracking_frame_scale.set(self._tracker.tracking_max_skip)
        self.tracking_frame_scale.pack(side=tk.TOP, fill=tk.X, pady=1, padx=5)

        display_options_frame = tk.LabelFrame(self.left_toolbox_frame, text="Display")
        display_options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)

        self.draw_trajectory_checkbox = tk.Checkbutton(display_options_frame, text="Draw Trajectory", variable=self._draw_trajectory)
        self.draw_trajectory_checkbox.pack(side=tk.TOP, anchor=tk.W, pady=2, padx=5)

        # Center pane - video display
        self.video_canvas = VideoCanvas(self.center_video_frame)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Right pane - tracking listbox
        self.scrollbar = tk.Scrollbar(self.right_listbox_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tracking_listbox = tk.Listbox(self.right_listbox_frame, yscrollcommand=self.scrollbar.set)
        self.tracking_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.tracking_listbox.yview)
        self.tracking_listbox.bind('<<ListboxSelect>>', self._listbox_item_selected)
        self.tracking_listbox.bind('<Delete>', self._listbox_item_deleted)

        # Bottom controls - track bar and timeline navigation.
        video_controls_frame = tk.Frame(self.bottom_controls_frame)
        video_controls_frame.pack(side=tk.TOP, fill=tk.X, expand=True)

        self.slider = tk.Scale(video_controls_frame, from_=0, to=self._num_frames, orient=tk.HORIZONTAL, showvalue=0, command=self.slider_value_changed)
        self.slider.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=(0, 2))

        self.track_bar = TrackingBar(video_controls_frame, self._num_frames, height=10)
        self.track_bar.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=(0, 5))

        nav_time_frame = tk.Frame(video_controls_frame)
        nav_time_frame.pack(side=tk.TOP, fill=tk.X, expand=True)

        tk.Frame(nav_time_frame).pack(side=tk.LEFT, expand=True)
        self.prev_frame_button = tk.Button(nav_time_frame, text=u"\u23EE", command=self.previous_frame)
        self.prev_frame_button.pack(side=tk.LEFT, padx=2)

        self.play_button = tk.Button(nav_time_frame, text=u"\u25B6", command=self._start_tracking)
        self.play_button.pack(side=tk.LEFT, padx=2)

        self.pause_button = tk.Button(nav_time_frame, text=u"\u23F8", command=self._pause_tracking)
        self.pause_button.pack(side=tk.LEFT, padx=2)

        self.next_frame_button = tk.Button(nav_time_frame, text=u"\u23ED", command=self.next_frame)
        self.next_frame_button.pack(side=tk.LEFT, padx=2)

        self.label = tk.Label(nav_time_frame, text="00:00:00", width=10)
        self.label.pack(side=tk.LEFT, padx=5)

        tk.Frame(nav_time_frame).pack(side=tk.RIGHT, expand=True)

        self.status_bar = tk.Label(self.bottom_controls_frame, text="Not Calibrated", fg="red", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.distance_status_bar = tk.Label(self.bottom_controls_frame, text="Distance: 0.0 cm", fg="blue", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.distance_status_bar.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

    def _setup_keypress_events(self):
        """Set keypress events for navigation."""
        self.bind("<Left>", lambda _: self._previous_tracked_frame())
        self.bind("<Right>", lambda _: self._next_tracked_frame())
        self.bind("<space>", lambda _: self._start_tracking())
        self.bind("<Escape>", lambda _: self._pause_tracking())
        self.bind("<Delete>", lambda _: self._delete_current_tracked_frame())
    
    def _setup_image_mouse_events(self):
        """Set mouse events for image interactions."""
        self.video_canvas.on_left_click_callback = self._handle_canvas_left_click
        self.video_canvas.on_right_click_callback = self._handle_canvas_right_click
        self.video_canvas.on_mouse_wheel_callback = self._seek_frames

    def _delete_current_tracked_frame(self):
        """Delete the currently tracked frame from the list."""
        current_frame = self._get_current_frame_number()
        if current_frame in self._fish_tracking:
            del self._fish_tracking[current_frame]
        
        self.update_smooth_trajectory()
        self._rebuild_tracking_listbox()
    
    def _rebuild_tracking_listbox(self):
        """Rebuild the tracking listbox from the current fish tracking data."""
        self.tracking_listbox.delete(0, tk.END)
        for frame_id, (_, confidence) in sorted(self._fish_tracking.items()):
            self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))
        self.update_gui()
    
    def update_smooth_trajectory(self):
        trajectory = np.array([ [t, coordinates[0], coordinates[1]] for t, (coordinates, _) in self._fish_tracking.items() ])
        self._smooth_trajectory = smooth_trajectory(trajectory, self._trajectory_smooth_size)

    def _handle_canvas_left_click(self, mapped_coords, event):
        if self._calibration_running:
            if mapped_coords:
                self._screen_points.append(mapped_coords)

                if len(self._screen_points) == 4:
                    self._do_calibration()

                self.update_gui()
        elif mapped_coords:
            current_frame = self._get_current_frame_number()
            self.insert_tracking_point(current_frame, mapped_coords, 1.0)

    def _handle_canvas_right_click(self, mapped_coords, event):
        current_frame = self._get_current_frame_number()
        self.remove_tracking_point(current_frame)
    
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
    
    def _seek_frames(self, event):
        """Move to the tracked frame based on mouse wheel scroll."""
        if event.delta > 0:
            self.next_frame()
        else:
            self.previous_frame()
    
    def insert_tracking_point(self, frame_id, coordinates, confidence):
        """Insert a tracking point into the listbox."""
        self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))
        self._fish_tracking[frame_id] = (coordinates, confidence)
        self.track_bar.mark_processed(frame_id)
        self.update_smooth_trajectory()
        self.update_gui()
    
    def remove_tracking_point(self, frame_id):
        """Remove a tracking point from the listbox."""
        if frame_id in self._fish_tracking:
            del self._fish_tracking[frame_id]
            self.track_bar.mark_not_processed(frame_id)
            self.update_smooth_trajectory()
            self._rebuild_tracking_listbox()
            self.update_gui()

    def _create_menu(self):
        """Set up the application menu bar."""
        self.menu_bar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load Video", command=self._load_video)
        self.file_menu.add_command(label="Load Tracking", command=self._load_tracking)
        self.file_menu.add_command(label="Save Tracking", command=self._save_tracking)
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
        self._tracker.tracking_threshold = float(value) / 100.0

    def _detection_threshold_changed(self, value):
        """Update the detection threshold for the fish tracker."""
        self._tracker.detection_threshold = float(value) / 100.0

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

        self.update_smooth_trajectory()
        self.update_gui()

    def _trajectory_frame_skip_changed(self, value):
        """Update the frame skip value used during tracking."""
        self._tracker.tracking_max_skip = int(value)
    
    def _tracking_model_register(self, frame, centroid, confidence):
        """Register a tracking model for the current frame."""
        (x, y) = self._scene_calibration.rectify_point(centroid)
        self.insert_tracking_point(frame, (int(x), int(y)), confidence)
    
    def _tracking_ui_update(self, frame):
        """Update the UI with the current frame."""
        self.update()

    def _run_tracking(self):
        """Run tracking on the selected frames."""

        if self._tracker is None:
            messagebox.showerror("Error", "Tracking model not initialized.")
            return
        if self.track_bar.processing_start is None or self.track_bar.processing_end is None:
            messagebox.showerror("Error", "Please set the processing start and end frames.")
            return

        start_frame = self.track_bar.processing_start
        end_frame = self.track_bar.processing_end
        progress_reporter = ProgressReporter(self, start_frame, end_frame)

        self._current_frame = self._read_current_frame()

        # Blur heavily the display image for background computation (focus) effect.

        self._current_frame = cv2.resize(self._current_frame, None, fx=0.1, fy=0.1)
        self._current_frame = cv2.GaussianBlur(self._current_frame, (3, 3), 0)
        self._current_frame = cv2.resize(self._current_frame, None, fx=10, fy=10)

        self.update_gui()

        try:
            self._tracker.run_tracking(
                self._clip,
                start_frame,
                end_frame,
                progress_reporter=progress_reporter,
                model_track_register=self._tracking_model_register,
                ui_update=self._tracking_ui_update)
        except Exception as e:
            messagebox.showerror("Error", f"Tracking failed: {e}")
            self.status_bar['text'] = "Tracking failed."
            self.status_bar['fg'] = "red"

        progress_reporter.close()

        # Redraw the current frame with the trajectory.
        self._current_frame = self._read_current_frame()
        self.update_gui()

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
        self._smooth_trajectory = None
        self.tracking_listbox.delete(0, tk.END)
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
        
        self.track_bar.clear()
        self.tracking_listbox.delete(0, tk.END)
        self._fish_tracking = {}

        for item in file_dict["tracking"]:
            frame_id = item["frame_id"]
            coordinates = tuple(item["coordinates"])
            confidence = item["confidence"]
            self._fish_tracking[frame_id] = (coordinates, confidence)
            self.track_bar.mark_processed(frame_id)
            self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, confidence))
        
        self.update_smooth_trajectory()
        self._rebuild_tracking_listbox()
        self.update_gui()

        self.status_bar['text'] = "Tracking data loaded from {}".format(filename)
        self.status_bar['fg'] = "green"

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
            "video": self._clip.path,
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

    def _load_scene_reference(self):
        # TODO: Implement loading of a scene reference for calibration.
        pass

    def _load_video(self):
        """Load a new video file and reinitialize the video processing."""
        path = filedialog.askopenfilename()
        if path == "":
            messagebox.showerror("Error", "No video selected.")
            return

        try:
            self._clip = VideoClip(path)
            self._num_frames = int(self._clip.duration * self._clip.fps)
            self.track_bar.change_number_of_frames(self._num_frames)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            self._clip = None

        if self._clip is not None:
            self._playing = False
            self._current_frame = None

            self.slider.config(to=self._num_frames)
            self.slider.set(0)
            self._load_first_frame()
            self._reset_calibration()

            self.video_canvas.video_fps = self._clip.fps

    def _reset_calibration(self):
        """Reset the calibration state."""
        self._calibrated = False
        self._calibration_running = False
        self._pixel_to_cm_ratio = 1.0
        self._screen_points = []
        self.calibration_button['text'] = "Calibrate"
        self.calibration_button['fg'] = "red"

        # Rebind default mouse events.
        self.update_gui()

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
                self._reset_calibration()

                # Rebind default mouse events.
                self.update_gui()

                return

        self._calibrated = False
        self._calibration_running = True
        self._screen_points = []
        self.update_gui()
        self.status_bar['text'] = "Calibration started - left click to select corner points of the scene."
        self.status_bar['fg'] = "orange"
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

        self._reload_frame()
        self.update_gui()

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
        if self._clip is None:
            # return a black frame of 720p
            self._current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            self._display_image = self._current_frame.copy()
            self.update_gui()
            return self._current_frame

        frame = self._clip.get_frame(float(frame_number) / float(self._clip.fps))
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
            time.sleep(0.5 / self._clip.fps)
            self.after(1, self.next_frame)

    def update_gui(self):
        """
        Update the image display and the time label based on the current frame.
        """
        fps = self._clip.fps if self._clip else 30
        current_frame_id = self._get_current_frame_number()

        self.label['text'] = self._frame_to_time(float(current_frame_id), fps)

        # Determine overlay data
        calib_points = self._screen_points if self._calibration_running or (self._calibrated and not self._screen_points) else []
        track_point = self._fish_tracking.get(current_frame_id, [None])[0]

        # Update the data info for correct trajectory drawing about the frame.
        self.video_canvas.set_current_frame_id_for_trajectory(current_frame_id)

        # Set overlays on canvas
        self.video_canvas.set_calibration_points_overlay(calib_points)
        self.video_canvas.set_tracked_point_overlay(track_point)

        trajectory_overlay = self._smooth_trajectory if self._draw_trajectory.get() else None
        self.video_canvas.set_trajectory_overlay(trajectory_overlay)

        # Set the main image (this triggers redraw including overlays)
        self.video_canvas.set_image(self._current_frame)

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
