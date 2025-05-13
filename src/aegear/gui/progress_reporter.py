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
        self.is_cancelled = False # Flag to indicate if user cancelled

        # Build the UI
        self._build_window()

    def _build_window(self):
        """Create the progress window and widgets, center it, and make it modal."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Tracking Progress")

        # Make this window transient to its parent
        # This helps with window stacking and behavior (e.g., stays on top)
        self.window.transient(self.parent)

        self.label = tk.Label(self.window, text="Progress: 0%", width=50)
        self.label.pack(pady=(10,5), padx=10)

        self.bar = ttk.Progressbar(self.window, length=300)
        self.bar.pack(pady=10, padx=10)

        self.cancel_button = tk.Button(
            self.window, text="Cancel", command=self._handle_cancel
        )
        self.cancel_button.pack(pady=(5,10))

        # Prevent closing via the window manager's 'X' button directly
        # without handling cleanup (like grab_release).
        # Instead, make 'X' button also call _handle_cancel.
        self.window.protocol("WM_DELETE_WINDOW", self._handle_cancel)

        # Defer centering and grab_set until window is ready to be displayed
        # This ensures calculations are based on the window's actual size.
        self.window.after_idle(self._center_and_make_modal)

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
        Check if the progress window still exists AND cancel hasn't been pressed.
        """
        return self.window.winfo_exists() and not self.is_cancelled

    def close(self):
        """Destroy the progress window and release the grab."""
        if self.window.winfo_exists():
            self.window.grab_release()
            self.window.destroy()

    def reset(self):
        """Reset the start time."""
        self.t0 = time.time()

    def _center_and_make_modal(self):
        """Center the window over its parent and make it modal."""
        if not self.window.winfo_exists():
            return

        # Force window to update its size based on contents
        self.window.update_idletasks()

        # Get parent window's geometry
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Get progress window's geometry
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        # Calculate position to center the window
        position_x = parent_x + (parent_width // 2) - window_width
        position_y = parent_y + (parent_height // 2) - window_height

        # Set the geometry (position only, size is determined by contents)
        self.window.geometry(f"+{position_x}+{position_y}")

        # Make the window modal (grab input focus)
        # This must be done AFTER the window is visible or about to be.
        self.window.grab_set()

        # Lift the window to ensure it's on top, though transient and grab_set usually handle this.
        self.window.lift()

        # Set focus to the window itself or the cancel button
        self.cancel_button.focus_set()

    def _handle_cancel(self):
        """Handles the cancel action from button or window close."""
        self.is_cancelled = True # Set cancel flag for the tracking process to check
        self.close() # Close the window and release grab