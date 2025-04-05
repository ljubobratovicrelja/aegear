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

    @property
    def frame_width(self):
        """Compute the width of a single frame rectangle."""
        return self.winfo_reqwidth() / self.total_frames

    def mark_processing_start(self, frame_number):
        """
        Mark a frame as the start of the processing region.
        Shows an error if the selected start is after an already marked end.
        """
        if self.processing_end is not None and self.processing_end < frame_number:
            tk.messagebox.showerror("Error", "Processing start cannot be after processing end.")
            return

        if self.processing_start is not None:
            # Clear previous start marking.
            self.delete("processing_start")

        self.processing_start = frame_number

        self.create_rectangle(frame_number * self.frame_width, 0,
                              (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                              fill='purple', width=5, tags="processing_start")

    def unmark_processing_start(self):
        """Remove the processing start marker."""
        if self.processing_start is not None:
            self.delete("processing_start")
            self.processing_start = None

    def mark_processing_end(self, frame_number):
        """
        Mark a frame as the end of the processing region.
        Shows an error if the selected end is before an already marked start.
        """
        if self.processing_start is not None and self.processing_start > frame_number:
            tk.messagebox.showerror("Error", "Processing end cannot be before processing start.")
            return

        if self.processing_end is not None:
            # Clear previous end marking.
            self.delete("processing_end")

        self.processing_end = frame_number

        self.create_rectangle(frame_number * self.frame_width, 0,
                              (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                              fill='purple', width=5, tags="processing_end")

    def unmark_processing_end(self):
        """Remove the processing end marker."""
        if self.processing_end is not None:
            self.delete("processing_end")
            self.processing_end = None

    def mark_processed(self, frame_number):
        """
        Mark a frame as processed by drawing a green rectangle.
        """
        if frame_number not in self.processed_frames:
            self.processed_frames.append(frame_number)
            self.create_rectangle(frame_number * self.frame_width, 0,
                                  (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                                  fill='green', tags="processed_{}".format(frame_number))

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
        for frame in self.processed_frames:
            self.delete("processed_{}".format(frame))
        self.processed_frames = []

    def is_tracked(self, frame_number):
        """
        Check if a given frame has been marked as processed.
        :return: True if processed, False otherwise.
        """
        return frame_number in self.processed_frames
