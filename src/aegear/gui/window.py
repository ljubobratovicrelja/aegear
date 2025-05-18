import os
import json
import threading
import time
import bisect
from typing import Optional

# Third-party imports
import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

# Internal modules
from aegear.calibration import SceneCalibration
from aegear.trajectory import smooth_trajectory, detect_trajectory_outliers
from aegear.tracker import FishTracker

from aegear.gui.tracking_bar import TrackingBar
from aegear.gui.progress_reporter import ProgressReporter
from aegear.gui.video_canvas import VideoCanvas

from aegear.utils import resource_path
from aegear.video import VideoClip

# Constants
DEFAULT_CALIBRATION_FILE = resource_path("data/calibration.xml")
LOGO_FILE = resource_path("media/logo.png")
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
        self._sorted_tracked_frame_ids = []
        self._outlier_frames = []
        self._trajectory_smooth_size = 5
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
        self._load_aegear_logo()
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

        self.calibration_button = tk.Button(calib_frame, text="Calibrate", command=self._calibrate, state=tk.DISABLED)
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

        # --- Cleanup tools section ---
        cleanup_frame = tk.LabelFrame(self.left_toolbox_frame, text="Cleanup")
        cleanup_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)

        self.highlight_outliers_button = tk.Button(
            cleanup_frame,
            text="Highlight Outliers",
            command=self._highlight_outliers,
            state=tk.NORMAL
        )
        self.highlight_outliers_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        self.outlier_threshold_scale = tk.Scale(
            cleanup_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            label="Outlier Threshold",
        )
        self.outlier_threshold_scale.set(10.0)
        self.outlier_threshold_scale.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        nav_outlier_frame = tk.Frame(cleanup_frame)
        nav_outlier_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=2, padx=5)

        self.prev_outlier_button = tk.Button(nav_outlier_frame, text="Previous Outlier", command=self._goto_previous_outlier)
        self.prev_outlier_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        self.next_outlier_button = tk.Button(nav_outlier_frame, text="Next Outlier", command=self._goto_next_outlier)
        self.next_outlier_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))

        self.delete_outliers_button = tk.Button(
            cleanup_frame,
            text="Delete Current Outliers",
            command=self._delete_current_outliers,
            state=tk.NORMAL
        )
        self.delete_outliers_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        self.delete_all_outliers_button = tk.Button(
            cleanup_frame,
            text="Delete All Outliers",
            command=self._delete_all_outliers,
            state=tk.NORMAL
        )
        self.delete_all_outliers_button.pack(side=tk.TOP, fill=tk.X, pady=2, padx=5)

        # --- Video Information section ---
        video_info_frame = tk.LabelFrame(self.left_toolbox_frame, text="Video Information")
        video_info_frame.pack(side=tk.TOP, fill=tk.X, expand=False, **toolbox_padding)
        self.video_info_labels = {}
        info_fields = [
            ("Filename", "filename"),
            ("FPS", "fps"),
            ("Resolution", "resolution"),
            ("Length (s)", "length"),
            ("Frames", "frames"),
        ]
        for label, key in info_fields:
            row = tk.Frame(video_info_frame)
            row.pack(side=tk.TOP, fill=tk.X, expand=False, pady=1, padx=2)
            tk.Label(row, text=label+":", anchor=tk.W, width=12).pack(side=tk.LEFT)
            val_label = tk.Label(row, text="-", anchor=tk.W)
            val_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.video_info_labels[key] = val_label
        self._update_video_info()

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
        self._tree_columns = ("frame", "centroid", "confidence")
        self._tree_column_headers = {"frame": "FRAME", "centroid": "CENTROID", "confidence": "CONFIDENCE"}
        self._tree_col_widths = {"frame": 80, "centroid": 150, "confidence": 100}
        self._tree_font = ("Courier", 10)

        # Create a style for the Treeview
        style = ttk.Style(self)
        style.configure("Custom.Treeview", font=self._tree_font, rowheight=int(self._tree_font[1] * 1.5))
        style.configure("Custom.Treeview.Heading", font=(self._tree_font[0], self._tree_font[1], 'bold'))
        # Add outlier row style (yellow background)
        style.map("Outlier.Treeview",
                  background=[('selected', '#ffe066'), ('!selected', '#fff9c4')])
        style.configure("Outlier.Treeview", background="#fff9c4")

        self.tracking_tree = ttk.Treeview(
            self.right_listbox_frame,
            columns=self._tree_columns,
            show="headings",
            style="Custom.Treeview"
        )

        # Define headings and column properties
        for col_id in self._tree_columns:
            self.tracking_tree.heading(
                col_id,
                text=self._tree_column_headers[col_id],
                anchor=tk.CENTER # Center the header text
            )
            self.tracking_tree.column(
                col_id,
                width=self._tree_col_widths[col_id],
                minwidth=50,
                anchor=tk.CENTER,
                stretch=tk.YES
            )

        # Add a vertical scrollbar
        self.tree_scrollbar_y = ttk.Scrollbar(
            self.right_listbox_frame,
            orient=tk.VERTICAL,
            command=self.tracking_tree.yview
        )
        self.tracking_tree.configure(yscrollcommand=self.tree_scrollbar_y.set)

        # Pack the Treeview and Scrollbar
        # The Treeview should take up most space, scrollbar next to it.
        self.tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tracking_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tracking_tree.bind('<<TreeviewSelect>>', self._treeview_item_selected)
        self.tracking_tree.bind('<Delete>', self._treeview_item_deleted) # Or handle deletion differently

        # Tag config for outlier rows
        self.tracking_tree.tag_configure('outlier', background='#fff9c4')

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

        self.timeline_label = tk.Label(nav_time_frame, text="00:00:00", width=50)
        self.timeline_label.pack(side=tk.LEFT, padx=5)

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
        self.remove_tracking_point(self._get_current_frame_number())

    def _rebuild_tracking_treeview(self):
        self.tracking_tree.delete(*self.tracking_tree.get_children())
        sorted_items = sorted(self._fish_tracking.items())
        treeview_values = [self._format_treeview_values(frame_id, coordinates, confidence)
                           for frame_id, (coordinates, confidence) in sorted_items]
        for idx, (frame_id, (coordinates, confidence)) in enumerate(sorted_items):
            item_iid = str(frame_id)
            tags = ()
            if frame_id in self._outlier_frames:
                tags = ('outlier',)
            self.tracking_tree.insert(
                parent='',
                index=idx,
                iid=item_iid,
                values=treeview_values[idx],
                tags=tags
            )
        self.update_gui()

    def _format_treeview_values(self, frame_id: int, coordinates: tuple[int, int], confidence: float) -> tuple[str, str, str]:
        """Helper to format data specifically for the Treeview values tuple."""
        frame_val_str = str(frame_id) # Frame ID for the first column

        if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
            # Consistent formatting for centroid
            centroid_val_str = f"({coordinates[0]:3d},{coordinates[1]:3d})"
        else:
            centroid_val_str = str(coordinates) # Fallback

        conf_val_str = f"{confidence * 100.0:.1f}%"
        return (frame_val_str, centroid_val_str, conf_val_str)


    def _register_tracking_to_ui(self, frame_id: int, coordinates: tuple[int, int], confidence: float) -> None:
        """
        Registers or updates a tracking point in the UI's Treeview.
        If frame_id exists, it updates the item.
        If not, it inserts the new item in an order sorted by frame_id.
        Optimized: If frame_id is greater than all current IDs, insert at the end; if less than all, insert at the beginning; otherwise, use bisect.
        """
        item_iid = str(frame_id)
        item_values = self._format_treeview_values(frame_id, coordinates, confidence)

        tags = ()
        if frame_id in self._outlier_frames:
            tags = ('outlier',)
        if self.tracking_tree.exists(item_iid):
            self.tracking_tree.item(item_iid, values=item_values, tags=tags)
        else:
            all_current_iids_str = self.tracking_tree.get_children('')
            if all_current_iids_str:
                try:
                    frame_ids = [int(iid) for iid in all_current_iids_str]
                except ValueError:
                    frame_ids = []
                if frame_ids:
                    min_id = min(frame_ids)
                    max_id = max(frame_ids)
                    if frame_id > max_id:
                        # Fast path: append at the end
                        self.tracking_tree.insert(
                            parent='',
                            index='end',
                            iid=item_iid,
                            values=item_values
                        )
                        self.tracking_tree.see(item_iid)
                        return
                    elif frame_id < min_id:
                        # Fast path: insert at the beginning
                        self.tracking_tree.insert(
                            parent='',
                            index=0,
                            iid=item_iid,
                            values=item_values
                        )
                        self.tracking_tree.see(item_iid)
                        return
                # Otherwise, do the bisect/insert as before
                frame_ids.sort()
                insertion_index_in_sorted_list = bisect.bisect_left(frame_ids, frame_id)
                self.tracking_tree.insert(
                    parent='',
                    index=insertion_index_in_sorted_list,
                    iid=item_iid,
                    values=item_values,
                    tags=tags
                )
            else:
                # Tree is empty, just insert
                self.tracking_tree.insert(
                    parent='',
                    index=0,
                    iid=item_iid,
                    values=item_values,
                    tags=tags
                )
        self.tracking_tree.see(item_iid)

    def _handle_canvas_left_click(self, mapped_coords, event):
        if self._calibrated:
            (x, y) = self._scene_calibration.unrectify_point(mapped_coords)
            mapped_coords = (int(x), int(y))

        if self._clip is None:
            # If no video, simply upon click launch video loading.
            return self._load_video()

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
        if self._clip is None:
            # If no video, don't interact.
            return

        current_frame = self._get_current_frame_number()
        self.remove_tracking_point(current_frame)

    def _update_treeview_selection(self, target_frame_id):
        """Clears current selection and selects/scrolls to the target_frame_id in Treeview."""
        if not hasattr(self, 'tracking_tree'):
            return

        target_iid = str(target_frame_id)

        for selected_item_iid in self.tracking_tree.selection():
            self.tracking_tree.selection_remove(selected_item_iid)

        if self.tracking_tree.exists(target_iid):
            self.tracking_tree.selection_set(target_iid)
            self.tracking_tree.see(target_iid) # Scroll to make the item visible
        else:
            # TODO: same as above, find the right way to report errors.
            print(f"WARNING: Frame ID {target_frame_id} not found in Treeview for selection.")

    def _next_tracked_frame(self):
        """Move to the next frame that has tracking data and update Treeview selection."""
        if not self._sorted_tracked_frame_ids:
            return

        current_frame_on_slider = self._get_current_frame_number()
        insertion_point = bisect.bisect_right(self._sorted_tracked_frame_ids, current_frame_on_slider)

        if insertion_point < len(self._sorted_tracked_frame_ids):
            next_tracked_frame_id = self._sorted_tracked_frame_ids[insertion_point]
            self._update_treeview_selection(next_tracked_frame_id)
        else:
            last_tracked_frame = self._sorted_tracked_frame_ids[-1]
            if current_frame_on_slider < last_tracked_frame:
                self._update_treeview_selection(last_tracked_frame)

    def _previous_tracked_frame(self):
        """Move to the previous frame that has tracking data and update Treeview selection."""
        if not self._sorted_tracked_frame_ids: # No tracked frames
            return

        current_frame_on_slider = self._get_current_frame_number()
        insertion_point = bisect.bisect_left(self._sorted_tracked_frame_ids, current_frame_on_slider)

        if insertion_point > 0:
            previous_tracked_frame_id = self._sorted_tracked_frame_ids[insertion_point - 1]

            if previous_tracked_frame_id == current_frame_on_slider and insertion_point - 2 >= 0:
                previous_tracked_frame_id = self._sorted_tracked_frame_ids[insertion_point - 2]
                self._update_treeview_selection(previous_tracked_frame_id)
            elif previous_tracked_frame_id < current_frame_on_slider:
                self._update_treeview_selection(previous_tracked_frame_id)
        else:
            first_tracked_frame = self._sorted_tracked_frame_ids[0]
            if current_frame_on_slider > first_tracked_frame:
                self._update_treeview_selection(first_tracked_frame)
    
    
    def _seek_frames(self, event: tk.Event) -> None:
        """Move to the tracked frame based on mouse wheel scroll."""
        if event.delta > 0:
            self.next_frame()
        else:
            self.previous_frame
    
    def updated_sorted_tracked_frame_ids(self) -> None:
        """Update the sorted list of tracked frame IDs."""
        self._sorted_tracked_frame_ids = sorted(self._fish_tracking.keys())
    
    def insert_tracking_point(self, frame_id: int, coordinates: tuple[int, int], confidence: float) -> None:
        """Insert a tracking point into the listbox."""
        self._fish_tracking[frame_id] = (coordinates, confidence)
        self.updated_sorted_tracked_frame_ids()
        self.track_bar.mark_processed(frame_id)
        self._register_tracking_to_ui(frame_id, coordinates, confidence)
        self.update_gui()
    
    def remove_tracking_point(self, frame_id: int) -> None:
        """Remove a tracking point from the listbox."""
        if frame_id in self._fish_tracking:
            del self._fish_tracking[frame_id]
            self.updated_sorted_tracked_frame_ids()
            self.track_bar.mark_not_processed(frame_id)
            item_iid = str(frame_id)
            if self.tracking_tree.exists(item_iid):
                self.tracking_tree.delete(item_iid)
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

    def _treeview_item_selected(self, event):
        """
        Callback when an item in the tracking listbox is selected.
        Updates the current frame to the selected frame.
        """
        selected_items = self.tracking_tree.selection() # Returns a tuple of selected item IDs
        if not selected_items:
            return

        item_iid = selected_items[0] # Assuming single selection
        # The iid is the frame_id we used during insert
        try:
            frame_id = int(item_iid)
            self._set_frame(frame_id) # Your method to jump to a frame
            self.update_gui()
        except ValueError:
            print(f"Error: Could not parse frame_id from Treeview item iid: {item_iid}")

    def _treeview_item_deleted(self, event):
        """
        Callback when an item in the tracking listbox is deleted.
        Removes the corresponding processed frame marker.
        """
        selected_items = self.tracking_tree.selection()
        if not selected_items:
            return

        for item_iid in selected_items:
            try:
                frame_id = int(item_iid)
                self.remove_tracking_point(frame_id)

            except ValueError:
                print(f"Error: Could not parse frame_id for deletion: {item_iid}")
                
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

        self.update_gui()

    def _trajectory_frame_skip_changed(self, value):
        """Update the frame skip value used during tracking."""
        self._tracker.tracking_max_skip = int(value)
    
    def _tracking_model_register(self, frame, centroid, confidence):
        """Register a tracking model for the current frame."""
        (x, y) = centroid
        coordinates = (int(x), int(y))
        self._fish_tracking[frame] = (coordinates, confidence)
    
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
                model_track_register=self._tracking_model_register,
                progress_reporter=progress_reporter,
                ui_update=self._tracking_ui_update)
        except Exception as e:
            messagebox.showerror("Error", f"Tracking failed: {e}")
            self.status_bar['text'] = "Tracking failed."
            self.status_bar['fg'] = "red"

        progress_reporter.close()

        # Refresh helper data after tracking
        self.updated_sorted_tracked_frame_ids()

        for frame, (coordinates, confidence) in self._fish_tracking.items():
            self._register_tracking_to_ui(frame, coordinates, confidence)
            self.track_bar.mark_processed(frame)

        self.update_gui()

        # Redraw the current frame with the trajectory.
        self._current_frame = self._read_current_frame()
        self.update_gui()

    def _set_track_start(self):
        """Set the current slider position as the start of processing."""
        self.track_bar.mark_processing_start(self.slider.get())
        self.update_gui()

    def _set_track_end(self):
        """Set the current slider position as the end of processing."""
        self.track_bar.mark_processing_end(self.slider.get())
        self.update_gui()

    def _reset_tracking(self):
        """Reset all tracking markers and clear the tracking list."""
        self.track_bar.clear()
        self._fish_tracking = {}
        self._sorted_tracked_frame_ids = []
        self.tracking_tree.delete(*self.tracking_tree.get_children())
        self._update_video_info()
        self.update_gui()

    def _about(self):
        """Display information about the application."""
        messagebox.showinfo("About", "Aegear\n\nAuthor: Relja Ljubobratovic\n\nGitHub: https://github.com/ljubobratovicrelja/aegear\nEmail: ljubobratovic.relja@gmail.com")

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
        
        self._reset_tracking()

        for item in file_dict["tracking"]:
            frame_id = item["frame_id"]
            coordinates = tuple(item["coordinates"])
            confidence = item["confidence"]
            self._fish_tracking[frame_id] = (coordinates, confidence)
            self.track_bar.mark_processed(frame_id)
            self._register_tracking_to_ui(frame_id, coordinates, confidence)
        
        self.updated_sorted_tracked_frame_ids()
        self._rebuild_tracking_treeview()
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

        # Note that we store the raw coordinate before calibration, so that the calibration can be reapplied later.
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
        if len(self._fish_tracking) > 0:
            if not messagebox.askokcancel("Warning", "Loading a new video will reset all tracking data. Proceed?"):
                return

        path = filedialog.askopenfilename()
        if path == "":
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
            self._reset_calibration()
            self._reset_tracking()

            self.video_canvas.video_fps = self._clip.fps

            # Enable calibration button in case it hasn't been done yet.
            self.calibration_button['state'] = tk.NORMAL

            self._reload_frame()
            self._update_video_info()

    def _reset_calibration(self):
        """Reset the calibration state."""
        self._calibrated = False
        self._calibration_running = False
        self._pixel_to_cm_ratio = 1.0
        self._screen_points = []
        self.calibration_button['text'] = "Calibrate"

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
        self.status_bar['text'] = "Calibration started - left click to select corner points of the scene."
        self.status_bar['fg'] = "orange"
        self.calibration_button['text'] = "Cancel Calibration"

        self.update_gui()

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
            self.set_track_start_button["state"] = tk.NORMAL
            self.set_track_end_button["state"] = tk.NORMAL
            self.run_tracking_button["state"] = tk.NORMAL
        except Exception as e:
            self._calibrated = False
            self._calibration_running = False
            self.status_bar['text'] = "Calibration failed - internal error: {}".format(e)
            self.status_bar['fg'] = "red"
            self.calibration_button['text'] = "Calibrate"

        self._screen_points = []

        self.updated_sorted_tracked_frame_ids()

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

            self._current_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Load logo PNG
            logo_img = cv2.imread(resource_path("media/logo.png"), cv2.IMREAD_UNCHANGED)
            if logo_img is not None:
                logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGRA2RGBA)
                h, w = logo_img.shape[:2]
                bh, bw = self._current_frame.shape[:2]

                x = (bw - w) // 2
                y = (bh - h) // 2

                overlay_rgb = logo_img[..., :3]
                alpha_mask =logo_img[..., 3] / 255.0

                roi = self._current_frame[y:y+h, x:x+w, :]

                blended = (roi * (1 - alpha_mask[..., np.newaxis]) + overlay_rgb * alpha_mask[..., np.newaxis]).astype(np.uint8)
                self._current_frame[y:y+h, x:x+w] = blended

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
    
    def _tracked_frames(self):
        """
        Get the tracked frames from the tracking data.
        Returns a list of frame IDs that have tracking data.
        """
        yield from self._sorted_tracked_frame_ids
    
    def _get_track_point(self, frame_id):
        """
        Get the tracking point for a specific frame, with calibration applied if any.
        Returns None if no tracking data is available for that frame.
        """
        if frame_id in self._fish_tracking:
            point = self._fish_tracking[frame_id][0]
            if self._calibrated:
                point = self._scene_calibration.rectify_point(point)
            return point
        return None
    
    def _get_tracking_data(self) -> Optional[np.ndarray]:
        """Get all present tracking data as a numpy array of coordinates, if present. If not, return None."""
        if self._fish_tracking:
            return np.array([self._fish_tracking[frame_id][0] for frame_id in self._sorted_tracked_frame_ids])
        return None
    
    def _get_current_track_point(self) -> Optional[tuple[int, int]]:
        """Get the tracking point for the current frame. Returns None if no tracking data is available for that frame."""
        current_frame_id = self._get_current_frame_number()
        return self._get_track_point(current_frame_id)
    
    def _compute_travel_distance(self) -> float:
        """Compute the travel distance based on the tracked points. Returns the total distance traveled in cm."""
        if self._fish_tracking:
            tracked_points = np.array([self._fish_tracking[frame_id][0] for frame_id in self._sorted_tracked_frame_ids])
            distances = np.linalg.norm(np.diff(tracked_points, axis=0), axis=1)
            return float(np.sum(distances) * self._pixel_to_cm_ratio)
        return 0.0

    def _compute_trajectory_overlay(
        self,
        current_frame_id: int,
        fps: float
    ) -> list[list[int]] | None:
        """
        Compute the smoothed trajectory overlay for the current frame window.
        Returns a list of [frame_id, x, y] or None if not to be drawn or not enough points.
        """
        window_seconds = self.video_canvas.trajectory_fade_seconds if hasattr(self, 'video_canvas') else 3.0
        window_frames = int(round(window_seconds * fps))
        all_traj = []
        for t in self._tracked_frames():
            pt = self._get_track_point(t)
            if pt is not None:
                x, y = pt
                all_traj.append([t, x, y])
        # Require at least 2 points to draw a trajectory
        if len(all_traj) < 2:
            return None
        time_stamps = [p[0] for p in all_traj]
        t_min = current_frame_id - window_frames
        start_idx = bisect.bisect_left(time_stamps, t_min)
        window_chunk = all_traj[start_idx:]
        if len(window_chunk) >= self._trajectory_smooth_size:
            smoothed_chunk = smooth_trajectory(window_chunk, self._trajectory_smooth_size)
        else:
            smoothed_chunk = window_chunk
        # Still require at least 2 points in the visible chunk
        if not (self._draw_trajectory.get() and smoothed_chunk and len(smoothed_chunk) >= 2):
            return None
        return smoothed_chunk

    def update_gui(self) -> None:
        """
        Update the image display and the time label based on the current frame.
        """
        fps = self._clip.fps if self._clip else 30
        current_frame_id = self._get_current_frame_number()

        self.timeline_label['text'] = f"Time: {self._frame_to_time(float(current_frame_id), fps)}, Frame: {current_frame_id}/{self._num_frames}"

        if self._clip and self.track_bar.processing_start is not None and self.track_bar.processing_end is not None:
            self.timeline_label['text'] += f", Tracking: {self.track_bar.processing_start} - {self.track_bar.processing_end}"
        else:
            self.timeline_label['text'] += ", Tracking: Not set"

        # Determine overlay data
        calib_points = self._screen_points if self._calibration_running or (self._calibrated and not self._screen_points) else []
        track_point = self._get_current_track_point()

        # Compute trajectory overlay using helper
        trajectory_overlay = self._compute_trajectory_overlay(current_frame_id, fps)
        self.video_canvas.set_trajectory_overlay(trajectory_overlay)

        # Update the trajectory length and distance status bar (use all points, not just window)
        travel_distance = self._compute_travel_distance()
        self.distance_status_bar['text'] = "Distance: {} cm".format(travel_distance)

        # Update the data info for correct trajectory drawing about the frame.
        self.video_canvas.set_current_frame_id_for_trajectory(current_frame_id)

        # Set overlays on canvas
        self.video_canvas.set_calibration_points_overlay(calib_points)
        self.video_canvas.set_tracked_point_overlay(track_point)

        # Set the main image (this triggers redraw including overlays)
        self.video_canvas.set_image(self._current_frame)

    def slider_value_changed(self, value: int) -> None:
        """Callback for slider value changes. Reloads and displays the corresponding frame."""
        self._current_frame = self._read_frame(int(value))
        self._display_image = self._current_frame.copy()
        self.update_gui()

    def _load_aegear_logo(self):
        """
        Load the first frame from the video.
        Also adjusts the tracking listbox height based on the frame size.
        """

        self._current_frame = self._read_frame(0)
        self._display_image = self._current_frame.copy()
        assert self._current_frame is not None, "Failed to load first frame."
        self._image_width = self._current_frame.shape[1]

    def _frame_to_time(self, frame: float, fps: float) -> str:
        """
        Convert a frame number to a formatted time string (hh:mm:ss).
        """
        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)
    
    def _detect_outliers(self):
        """
        Detect outliers in the trajectory using the specified threshold.
        Returns a list of outlier frame IDs.
        """
        # Prepare trajectory as list of (frame, x, y)
        trajectory = [
            (frame_id, coords[0][0], coords[0][1])
            for frame_id, coords in self._fish_tracking.items()
        ]

        if len(trajectory) < 3:
            return None

        threshold = float(self.outlier_threshold_scale.get())
        return set(detect_trajectory_outliers(trajectory, threshold))

    def _highlight_outliers(self, show_message=True):
        """
        Run outlier detection on the trajectory and highlight outlier frames in the Treeview.
        """

        self._outlier_frames = self._detect_outliers()

        if not self._outlier_frames and show_message:
            messagebox.showinfo("Highlight Outliers", "No outliers detected with current settings.")
            return

        # Update tags for all items
        for frame_id in self._fish_tracking:
            iid = str(frame_id)
            tags = ()
            if frame_id in self._outlier_frames:
                tags = ('outlier',)
            if self.tracking_tree.exists(iid):
                self.tracking_tree.item(iid, tags=tags)

        if show_message:
            messagebox.showinfo("Highlight Outliers", f"Highlighted {len(self._outlier_frames)} outlier(s) in yellow.")

    def _goto_next_outlier(self):
        """Go to the next outlier frame after the current frame."""
        if not self._outlier_frames:
            return
        current_frame = self._get_current_frame_number()
        outlier_ids = sorted(self._outlier_frames)
        for frame_id in outlier_ids:
            if frame_id > current_frame:
                self._set_frame(frame_id)
                self._update_treeview_selection(frame_id)
                return
        # If none found, wrap to first
        self._set_frame(outlier_ids[0])
        self._update_treeview_selection(outlier_ids[0])

    def _goto_previous_outlier(self):
        """Go to the previous outlier frame before the current frame."""
        if not self._outlier_frames:
            return
        current_frame = self._get_current_frame_number()
        outlier_ids = sorted(self._outlier_frames)
        for frame_id in reversed(outlier_ids):
            if frame_id < current_frame:
                self._set_frame(frame_id)
                self._update_treeview_selection(frame_id)
                return
        # If none found, wrap to last
        self._set_frame(outlier_ids[-1])
        self._update_treeview_selection(outlier_ids[-1])

    def _delete_current_outliers(self, show_message=True):
        """Delete all tracking points currently marked as outliers."""
        if not self._outlier_frames and show_message:
            messagebox.showinfo("Delete Outliers", "No outliers to delete.")
            return
        count = 0
        for frame_id in list(self._outlier_frames):
            if frame_id in self._fish_tracking:
                del self._fish_tracking[frame_id]
                count += 1

        self._outlier_frames = set()

        self.updated_sorted_tracked_frame_ids()
        self._rebuild_tracking_treeview()
        self.update_gui()

        if show_message:
            messagebox.showinfo("Delete Outliers", f"Deleted {count} outlier(s) from tracking data.")

    def _delete_all_outliers(self):
        """
        Iteratively detect and delete outliers for the current threshold until none remain.
        """
        threshold = self.outlier_threshold_scale.get()
        if not messagebox.askokcancel(
            "Delete All Outliers",
            f"Are you sure you want to delete ALL outliers? This will remove all detected outliers for threshold {threshold}. This action cannot be undone."):
            return

        num_deleted = 0

        while True:
            # Detect outliers
            outliers = self._detect_outliers()
            
            if not outliers:
                break

            deleted_this_iteration = len(outliers)

            for frame_id in outliers:
                del self._fish_tracking[frame_id]
                self.track_bar.mark_not_processed(frame_id)

            num_deleted += deleted_this_iteration

        self.updated_sorted_tracked_frame_ids()
        self._rebuild_tracking_treeview()
        self.update_gui()

        self.status_bar['text'] = f"{num_deleted} outliers deleted for threshold {threshold}."
        self.status_bar['fg'] = "green"

    def _update_video_info(self):
        """Update the video information section with current video info."""
        if self._clip is not None:
            filename = getattr(self._clip, 'path', '-')
            fps = getattr(self._clip, 'fps', '-')
            width = getattr(self._clip, 'width', None)
            height = getattr(self._clip, 'height', None)
            if width is None or height is None:
                # Try to get from first frame
                try:
                    frame0 = self._clip.get_frame(0)
                    height, width = frame0.shape[:2]
                except Exception:
                    width = height = '-'
            resolution = f"{width} x {height}" if width != '-' and height != '-' else '-'
            frames = int(self._clip.duration * self._clip.fps) if self._clip.duration and self._clip.fps else '-'
            length = self._frame_to_time(frames, fps)
        else:
            filename = fps = resolution = length = frames = '-'
        
        self.video_info_labels["filename"].config(text=str(os.path.basename(filename)))
        self.video_info_labels["fps"].config(text=str(fps))
        self.video_info_labels["resolution"].config(text=str(resolution))
        self.video_info_labels["length"].config(text=str(length))
        self.video_info_labels["frames"].config(text=str(frames))
