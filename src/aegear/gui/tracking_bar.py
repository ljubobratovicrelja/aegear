import tkinter as tk


class TrackingBar(tk.Canvas):
    """
    A canvas widget for displaying video frame tracking information.
    It marks processed frames, and indicates start and end frames for processing.
    """

    def __init__(self, parent, frames, **kwargs):
        """
        :param parent: Parent tkinter widget.
        :param frames: Total number of frames in the video.
        """
        super(TrackingBar, self).__init__(parent, bg='gray', **kwargs)
        self.total_frames = frames
        self.processed_frames = []
        self.processing_start = None
        self.processing_end = None

        # Bind to <Configure> to handle resizes
        self.bind("<Configure>", self._on_resize)

    @property
    def frame_width(self):
        """Compute the width of a single frame rectangle based on actual canvas width."""
        actual_width = self.winfo_width()
        if self.total_frames > 0 and actual_width > 1:
            return actual_width / float(self.total_frames)
        return 1.0
    
    def change_number_of_frames(self, frames):
        """
        Change the number of frames in the video.
        :param frames: New total number of frames.
        """
        self.total_frames = frames
        self.processed_frames = []
        self.processing_start = None
        self.processing_end = None
        self.delete("all")

    def _get_center_x_for_frame(self, frame_number):
        """Calculate the logical X-coordinate matching the slider value (handle's leading edge)."""
        actual_width = self.winfo_width()
        if self.total_frames == 0 or actual_width <= 1:
            return actual_width / 2.0

        # This calculation seems to align with the handle's left edge based on image
        logical_center_x = (float(frame_number) + 0.5) / float(self.total_frames) * actual_width
        return logical_center_x

    def mark_processing_start(self, frame_number):
        """
        Mark a frame as the start of the processing region.
        Shows an error if the selected start is after an already marked end.
        """
        if self.processing_end is not None and self.processing_end < frame_number:
            tk.messagebox.showerror("Error", "Processing start cannot be after processing end.")
            return

        self.delete("processing_start")

        self.processing_start = frame_number
        actual_height = self.winfo_height()
        if actual_height <= 1: return

        center_x = self._get_center_x_for_frame(frame_number)
        marker_pixel_width = 4

        x1 = center_x - (marker_pixel_width / 2.0)
        x2 = center_x + (marker_pixel_width / 2.0)
        
        self.create_rectangle(x1, 0, x2, actual_height,
                              fill='black', outline='black',
                              tags="processing_start")

    def unmark_processing_start(self):
        if self.processing_start is not None:
            self.delete("processing_start")
            self.processing_start = None

    def mark_processing_end(self, frame_number):
        if self.processing_start is not None and self.processing_start > frame_number:
            tk.messagebox.showerror("Error", "Processing end cannot be before processing start.")
            return

        self.delete("processing_end")

        self.processing_end = frame_number
        actual_height = self.winfo_height()
        if actual_height <= 1: return

        center_x = self._get_center_x_for_frame(frame_number)
        marker_pixel_width = 4

        x1 = center_x - (marker_pixel_width / 2.0)
        x2 = center_x + (marker_pixel_width / 2.0)

        self.create_rectangle(x1, 0, x2, actual_height,
                              fill='black', outline='black',
                              tags="processing_end")

    def unmark_processing_end(self):
        """Remove the processing end marker."""
        if self.processing_end is not None:
            self.delete("processing_end")
            self.processing_end = None

    def mark_processed(self, frame_number):
        """
        Mark a frame as processed by drawing a green rectangle.
        """
        actual_height = self.winfo_height()
        if actual_height <= 1 or self.total_frames == 0: return # Not ready

        center_x = self._get_center_x_for_frame(frame_number)
        marker_pixel_width = 4

        x1 = center_x - (marker_pixel_width / 2.0)
        x2 = center_x + (marker_pixel_width / 2.0)
        
        if frame_number not in self.processed_frames:
             self.processed_frames.append(frame_number)

        self.create_rectangle(x1, 0, x2, actual_height,
                              fill='green', outline='green', # Solid green
                              tags="processed_{}".format(frame_number))


    def mark_not_processed(self, frame_number):
        """
        Remove the processed mark from a frame.
        """
        if frame_number not in self.processed_frames:
            return

        self.processed_frames.remove(frame_number)
        self.delete("processed_{}".format(frame_number))

    def clear(self):
        """Clear all processed marks."""
        self.delete("all")
        self.processing_start = None
        self.processing_end = None
        self.processed_frames = []

    def is_tracked(self, frame_number):
        """
        Check if a given frame has been marked as processed.
        :return: True if processed, False otherwise.
        """
        return frame_number in self.processed_frames

    def _on_resize(self, event=None):
        """Handles canvas resize to redraw markers accurately."""
        
        _start = self.processing_start
        _end = self.processing_end
        _processed_copy = list(self.processed_frames)

        # Clear the canvas for redrawing.
        self.clear()

        # Redraw based on stored state
        if _start is not None:
            self.mark_processing_start(_start)
        if _end is not None:
            self.mark_processing_end(_end)
        for frame_num in _processed_copy:
            self.mark_processed(frame_num)