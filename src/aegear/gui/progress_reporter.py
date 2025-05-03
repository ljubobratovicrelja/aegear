# progress_reporter.py

import time
import tkinter as tk
from tkinter import ttk


class ProgressReporter:
    """
    Tracks and reports the progress of a long-running task.

    Manages a separate window with a label and progress bar, showing
    percentage complete, ETA, and FPS.
    """

    def __init__(self, parent, start_frame, end_frame):
        """
        Initialize the progress reporter and build the UI window.

        Args:
            parent (tk.Widget): Parent widget (usually the main app).
            start_frame (int): Starting frame number.
            end_frame (int): Ending frame number.
        """
        self.parent = parent
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.total = end_frame - start_frame
        self.t0 = time.time()

        # Build the UI
        self._build_window()

    def _build_window(self):
        """Create the progress window and widgets."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Tracking")
        self.window.geometry("350x130")

        self.label = tk.Label(self.window, text="Progress: 0%")
        self.label.pack()

        self.bar = ttk.Progressbar(self.window, length=240)
        self.bar.pack(pady=15)

        tk.Button(
            self.window, text="Cancel", command=self.window.destroy
        ).pack()

    def update(self, current_frame):
        """
        Update the progress display based on the current frame.

        Args:
            current_frame (int): The frame number currently being processed.
        """
        done = current_frame - self.start_frame
        pct = (done / max(self.total, 1)) * 100.0
        elapsed = time.time() - self.t0

        eta_sec = (elapsed / max(pct, 0.001)) * (100.0 - pct)
        h, rem = divmod(int(eta_sec), 3600)
        m, s = divmod(rem, 60)

        video_fps = (done or 1) / max(elapsed, 1e-6)

        self.label["text"] = (
            f"Progress: {pct:5.1f}% | "
            f"ETA {h:02d}:{m:02d}:{s:02d} | "
            f"FPS {video_fps:5.1f}"
        )
        self.bar["value"] = pct

    def still_running(self):
        """
        Check if the progress window still exists (not closed by user).

        Returns:
            bool: True if the window still exists.
        """
        return self.window.winfo_exists()

    def close(self):
        """Destroy the progress window."""
        if self.window.winfo_exists():
            self.window.grab_set()
            self.window.destroy()

    def reset(self):
        """Reset the start time."""
        self.t0 = time.time()
