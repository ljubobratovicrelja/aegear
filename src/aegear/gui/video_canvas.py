import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import bisect
import numpy as np


class VideoCanvas(tk.Frame):
    """
    A custom Tkinter widget for displaying video frames with overlays.
    Handles image resizing, coordinate mapping, and drawing of visual elements.
    """
    def __init__(self,
                 master,
                 trajectory_fade_seconds: float = 3.0,
                 video_fps: float = 60.0,
                 **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg='black')

        # The Label widget where the image will be actually displayed
        self.image_label = tk.Label(self, bg='black', cursor="cross")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.source_frame_width = 1
        self.source_frame_height = 1
        self.current_display_scale = 1.0
        self.current_display_offset_x = 0.0
        self.current_display_offset_y = 0.0

        self._current_pil_image = None
        self._display_photo_image = None

        self.on_left_click_callback = None
        self.on_right_click_callback = None
        self.on_mouse_wheel_callback = None

        self.image_label.bind("<Button-1>", self._handle_left_click)
        self.image_label.bind("<Button-3>", self._handle_right_click)
        self.image_label.bind("<MouseWheel>", self._handle_mouse_wheel)
        self.bind("<Configure>", self._on_resize)

        self._calibration_points_to_draw = []
        self._tracked_point_to_draw = None
        self._trajectory_data = []
        self.current_frame_id_for_trajectory = 0
        self.trajectory_fade_seconds = trajectory_fade_seconds
        self.video_fps = video_fps
        self.trajectory_base_color = (255, 255, 0)
        self.trajectory_thickness = 2

    def _calculate_display_geometry(self):
        """
        Calculates the scaling factor and offsets for displaying the source image
        within the widget, maintaining aspect ratio.
        Returns True if geometry is successfully calculated, False otherwise.
        """
        widget_width = self.winfo_width()
        widget_height = self.winfo_height()

        # Check for invalid or uninitialized dimensions
        if widget_width <= 1 or widget_height <= 1 or \
           self.source_frame_width <= 0 or self.source_frame_height <= 0:
            self.current_display_scale = 1.0 # Default to no scaling
            self.current_display_offset_x = 0.0
            self.current_display_offset_y = 0.0
            return False

        # Calculate scale to fit, preserving aspect ratio
        scale_w = float(widget_width) / self.source_frame_width
        scale_h = float(widget_height) / self.source_frame_height
        self.current_display_scale = min(scale_w, scale_h)

        # Calculate dimensions of the image as it will be displayed
        display_width = int(self.source_frame_width * self.current_display_scale)
        display_height = int(self.source_frame_height * self.current_display_scale)

        # Calculate offsets for centering
        self.current_display_offset_x = (widget_width - display_width) / 2.0
        self.current_display_offset_y = (widget_height - display_height) / 2.0
        return True

    def set_image(self, numpy_frame):
        """
        Sets the current video frame to be displayed.
        numpy_frame: A NumPy array representing the image (e.g., from OpenCV).
        """
        if numpy_frame is None or numpy_frame.size == 0:
            self.image_label.config(image='') # Clear the label
            self._display_photo_image = None
            self._current_pil_image = None
            self.source_frame_width = 1 # Reset dimensions
            self.source_frame_height = 1
            return

        # Update source dimensions based on the new frame
        self.source_frame_height, self.source_frame_width = numpy_frame.shape[:2]
        self._current_pil_image = Image.fromarray(numpy_frame) # Convert to PIL
        self.redraw_image_with_overlays() # Redraw with the new image and existing overlays

    def redraw_image_with_overlays(self):
        """
        Redraws the current PIL image, applies all active overlays,
        resizes it according to current geometry, and updates the display.
        """
        if self._current_pil_image is None:
            self.image_label.config(image='')
            self._display_photo_image = None
            return

        # Attempt to calculate display geometry. If it fails (e.g., widget not sized),
        # it might draw unscaled or wait for a resize event.
        geometry_valid = self._calculate_display_geometry()

        # Create a working copy of the PIL image to draw overlays on
        image_for_display = self._current_pil_image.copy()

        # Draw all stored overlays onto this copy (using source coordinates)
        self._apply_overlays_to_pil_image(image_for_display)

        # Resize the PIL image (with overlays) if geometry is valid
        if geometry_valid:
            display_width = int(self.source_frame_width * self.current_display_scale)
            display_height = int(self.source_frame_height * self.current_display_scale)

            # Ensure calculated dimensions are positive before attempting resize
            if display_width > 0 and display_height > 0:
                try:
                    resampling_filter = Image.Resampling.BILINEAR if self.current_display_scale < 1.0 else Image.Resampling.NEAREST
                    resized_image_pil = image_for_display.resize((display_width, display_height), resampling_filter)
                except ValueError:
                    resized_image_pil = image_for_display # Fallback
            else:
                resized_image_pil = image_for_display
        else:
            resized_image_pil = image_for_display

        # Convert the final PIL image to PhotoImage and update the label
        self._display_photo_image = ImageTk.PhotoImage(resized_image_pil)
        self.image_label.configure(image=self._display_photo_image)

    def _apply_overlays_to_pil_image(self, pil_image_target):
        """
        Helper method to draw all currently defined overlays onto the provided PIL.Image object.
        Coordinates for drawing are assumed to be in the source image's coordinate system.
        """
        # Ensure pil_image_target is a valid PIL Image object
        if not isinstance(pil_image_target, Image.Image):
             print("Error: _apply_overlays_to_pil_image received non-PIL image.")
             return

        draw = ImageDraw.Draw(pil_image_target)
        radius = 5

        for point in self._calibration_points_to_draw:
            x, y = point
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline="blue", width=2)

        if self._tracked_point_to_draw:
            x, y = self._tracked_point_to_draw
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline="lime", width=2)

        window_frames = int(round(self.trajectory_fade_seconds * self.video_fps))

        if window_frames > 0 and self._trajectory_data and len(self._trajectory_data) >= 2:
            current_t = self.current_frame_id_for_trajectory
            t_min = current_t - window_frames

            time_stamps = [p[0] for p in self._trajectory_data]
            start_idx = bisect.bisect_left(time_stamps, t_min)

            if start_idx < len(self._trajectory_data) -1:
                pts_slice = self._trajectory_data[start_idx:]

                for i in range(1, len(pts_slice)):
                    t1, x1, y1 = pts_slice[i-1]
                    t2, x2, y2 = pts_slice[i]

                    age = current_t - t2
                    if age > window_frames or age < 0:
                        continue

                    alpha = max(0.0, 1.0 - (float(age) / window_frames))
                    
                    # TODO: temporary solution: see int the future if we can actually have alpha or
                    # nice gradient instead of just fading to black.
                    faded_color = tuple(int(c * alpha) for c in self.trajectory_base_color)
                    
                    draw.line([(x1, y1), (x2, y2)], fill=faded_color, width=self.trajectory_thickness)

    def get_mapped_coordinates(self, event_x, event_y):
        """
        Maps click coordinates from a Tkinter event (relative to the image_label)
        back to the coordinates of the original source image.
        Returns a tuple (source_x, source_y) or None if mapping is not possible
        (e.g., click outside image area, or geometry not calculated).
        """
        # Check if geometry is valid for mapping
        if self.current_display_scale <= 1e-6 or self.source_frame_width <= 0: # Scale can be very small
            return None

        # Coordinates of the click relative to the top-left of the displayed image area
        click_x_on_displayed_image = event_x - self.current_display_offset_x
        click_y_on_displayed_image = event_y - self.current_display_offset_y

        # Actual dimensions of the image as shown on the label
        displayed_image_width_on_label = self.source_frame_width * self.current_display_scale
        displayed_image_height_on_label = self.source_frame_height * self.current_display_scale

        # Check if the click was within the bounds of the actual displayed image
        if not (0 <= click_x_on_displayed_image < displayed_image_width_on_label and
                0 <= click_y_on_displayed_image < displayed_image_height_on_label):
            return None # Click was in the label's padding area

        # Scale back to source image coordinates
        source_x = click_x_on_displayed_image / self.current_display_scale
        source_y = click_y_on_displayed_image / self.current_display_scale

        # Clamp coordinates to be within the source frame bounds (due to potential float precision issues)
        source_x = max(0, min(self.source_frame_width - 1, source_x))
        source_y = max(0, min(self.source_frame_height - 1, source_y))

        return (int(round(source_x)), int(round(source_y)))

    def _on_resize(self, event=None):
        """Handles resize of the VideoCanvas Frame itself."""
        # When the frame resizes, the image needs to be redrawn
        # to fit the new dimensions.
        if self._current_pil_image: # Only redraw if an image is loaded
            self.redraw_image_with_overlays()

    def _handle_left_click(self, event):
        if self.on_left_click_callback:
            mapped_coords = self.get_mapped_coordinates(event.x, event.y)
            self.on_left_click_callback(mapped_coords, event)

    def _handle_right_click(self, event):
        if self.on_right_click_callback:
            mapped_coords = self.get_mapped_coordinates(event.x, event.y)
            self.on_right_click_callback(mapped_coords, event)

    def _handle_mouse_wheel(self, event):
        if self.on_mouse_wheel_callback:
            self.on_mouse_wheel_callback(event) # Pass the raw event

    def clear_all_overlays(self):
        self._calibration_points_to_draw = []
        self._tracked_point_to_draw = None
        self._trajectory_to_draw = []
        if self._current_pil_image:
            self.redraw_image_with_overlays()

    def set_calibration_points_overlay(self, points_list):
        """Sets the list of calibration points to draw.
        points_list: A list of (x,y) tuples in SOURCE coordinates.
        """
        self._calibration_points_to_draw = list(points_list) # Store a copy
        if self._current_pil_image:
            self.redraw_image_with_overlays()

    def set_tracked_point_overlay(self, point):
        """Sets the current tracked point to draw.
        point: An (x,y) tuple in SOURCE coordinates, or None to clear.
        """
        self._tracked_point_to_draw = point
        if self._current_pil_image:
            self.redraw_image_with_overlays()

    def set_current_frame_id_for_trajectory(self, frame_id: int):
        """Sets the current frame ID, used for time-dependent effects like trajectory fading."""
        self.current_frame_id_for_trajectory = frame_id

    def set_trajectory_overlay(self, trajectory_data_list):
        """Sets the trajectory path data.
        trajectory_data_list: A list of [time_index, x, y] items in SOURCE coordinates.
                              Assumed to be sorted by time_index.
        """
        if trajectory_data_list is not None:
            if isinstance(trajectory_data_list, np.ndarray):
                self._trajectory_data = trajectory_data_list.tolist()
            else:
                self._trajectory_data = list(trajectory_data_list) # Store a copy
        else:
            self._trajectory_data = []