import sys

# add the current directory to the path for maze modules
sys.path.append(".")

import threading
import time

import numpy as np

import cv2
from PIL import Image, ImageTk

import torch
from torchvision import transforms

import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog

from tkinter import ttk

from moviepy.editor import VideoFileClip

from mazecalibration import MazeCalibration
from motiondetection import MotionDetector

# needed for the classifier loading
import maze.classifier


class TrackingBar(tk.Canvas):
    def __init__(self, parent, frames, **kwargs):
        super(TrackingBar, self).__init__(parent, bg='gray', **kwargs)
        self.total_frames = frames
        self.processed_frames = []
        self.processing_start = None
        self.processing_end = None
    
    @property
    def frame_width(self):
        return self.winfo_reqwidth() / self.total_frames

    def mark_processing_start(self, frame_number):
        # check if end is before start
        if self.processing_end is not None and self.processing_end < frame_number:
            tk.messagebox.showerror("Error", "Processing start cannot be after processing end.")
            return

        if self.processing_start is not None:
            # clear the rectangle that was previously drawn
            self.delete("processing_start")
        
        
        self.processing_start = frame_number

        self.create_rectangle(frame_number * self.frame_width, 0,
                                (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                                fill='purple', width=5, tags="processing_start")
    
    def mark_processing_end(self, frame_number):
        # check if start is after end
        if self.processing_start is not None and self.processing_start > frame_number:
            tk.messagebox.showerror("Error", "Processing end cannot be before processing start.")
            return

        if self.processing_end is not None:
            # clear the rectangle that was previously drawn
            self.delete("processing_end")

        self.processing_end = frame_number

        self.create_rectangle(frame_number * self.frame_width, 0,
                                (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                                fill='purple', width=5, tags="processing_end")

    def mark_processed(self, frame_number):
        if frame_number not in self.processed_frames:
            self.processed_frames.append(frame_number)
            self.create_rectangle(frame_number * self.frame_width, 0,
                                  (frame_number + 1) * self.frame_width, self.winfo_reqheight(),
                                  fill='green', tags="processed_{}".format(frame_number))

    def mark_not_processed(self, frame_number):
        if frame_number not in self.processed_frames:
            return

        self.processed_frames.remove(frame_number)
        self.delete("processed_{}".format(frame_number))
    
    def clear(self):
        for frame in self.processed_frames:
            self.delete("processed_{}".format(frame))
        self.processed_frames = []
    
    def is_tracked(self, frame_number):
        return frame_number in self.processed_frames

class MainWindow(tk.Tk):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.withdraw()

        # set window Title
        self.title("Maze Tracking")

        # width of the image, to be updated upon first frame load
        self._current_frame = None
        self._display_image = None
        self._image_width = 0
        self._playing = False
        self._calibrated = False
        self._calibration_running = False
        self._pixel_to_cm_ratio = 1.0  # default to 1.0

        self._fish_tracking = {}

        self._classification_threshold = 0.85
        self._classifier_model = None

        # classifier transformations 
        self._transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert np array to PIL Image
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
        ])


        # screen points for calibration and other purposes
        self._screen_points = []

        # initiate maze calibration utility
        self._maze_calibration = MazeCalibration("data/calibration.xml")

        # motion detector
        self._motion_detector = MotionDetector(0.03, 12) # threshold and block size

        self.dialog_window = tk.Toplevel(self)
        self.dialog_window.withdraw()

        self.dialog_window.title("Load Video")

        #initial_video = filedialog.askopenfilename(parent=self.dialog_window)
        initial_video = "data/videos/K9.MOV"

        if initial_video == "":
            # warning dialog and close the app
            messagebox.showerror("Error", "No video selected.")
            self.destroy()
            sys.exit(1)

        # video clip to be loaded
        self.clip = VideoFileClip(initial_video)

        # initialize useful video metadata
        self._num_frames = self.clip.duration * self.clip.fps

        # setup frames - center for video, bottom for slider and right for tools
        self.center_frame = tk.Frame(self)
        self.center_frame.pack(side=tk.TOP)

        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side=tk.BOTTOM)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT)

        # Create a label to hold the image
        self.image_label = tk.Label(self.center_frame, cursor="cross")
        self.image_label.pack()

        self._load_first_frame()

        self.slider = tk.Scale(self.bottom_frame, from_=0, to=self._num_frames, length=self._image_width, orient=tk.HORIZONTAL, command=self.slider_value_changed)
        self.slider.pack()

        self.track_bar = TrackingBar(self.bottom_frame, self._num_frames, width=self._image_width, height=10)
        self.track_bar.pack()

        self.label = tk.Label(self.bottom_frame, text="00:00:00")
        self.label.pack()

        self.calibration_button = tk.Button(self.right_frame, text="Calibrate", command=self._calibrate, fg="red")
        self.calibration_button.pack(side=tk.LEFT)

        self.set_track_start_button = tk.Button(self.right_frame, text="Set Track Start", command=self._set_track_start)
        self.set_track_start_button.pack(side=tk.LEFT)

        self.set_track_end_button = tk.Button(self.right_frame, text="Set Track End", command=self._set_track_end)
        self.set_track_end_button.pack(side=tk.LEFT)

        self.run_tracking_button = tk.Button(self.right_frame, text="Run Tracking", command=self._run_tracking, state=tk.DISABLED)
        self.run_tracking_button.pack(side=tk.LEFT)

        self.reset_tracking_button = tk.Button(self.right_frame, text="Reset Tracking", command=self._reset_tracking)
        self.reset_tracking_button.pack(side=tk.LEFT)

        self.clear_tracking_frame_button = tk.Button(self.right_frame, text="Clear Frame Track", command=self._clear_tracking_frame)
        self.clear_tracking_frame_button .pack(side=tk.LEFT)

        self.motion_threshold_scale = tk.Scale(self.right_frame, from_=0.0, to=0.3, resolution=0.01, orient=tk.HORIZONTAL, command=self._motion_threshold_changed)
        self.motion_threshold_scale.set(0.03)
        self.motion_threshold_scale.pack(side=tk.LEFT)

        self.prev_frame_button = tk.Button(self.right_frame, text=u"\u23EE", command=self._prev_frame)
        self.prev_frame_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.right_frame, text=u"\u23F8", command=self._pause_tracking)
        self.pause_button.pack(side=tk.RIGHT)

        self.play_button = tk.Button(self.right_frame, text=u"\u25B6", command=self._start_tracking)
        self.play_button.pack(side=tk.RIGHT)

        self.next_frame_button = tk.Button(self.right_frame, text=u"\u23ED", command=self._next_frame)
        self.next_frame_button.pack(side=tk.RIGHT)

        # have a status bar at the bottom
        self.status_bar = tk.Label(self.bottom_frame, text="Not Calibrated", fg="red", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_gui()

        # create menu bar
        self.menu_bar = tk.Menu(self)

        # create file menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load Video", command=self._load_video)
        self.file_menu.add_command(label="Load Classifier Model", command=self._load_classifier)
        self.file_menu.add_command(label="Save Tracking", command=self._save_tracking)
        self.file_menu.add_command(label="Exit", command=self.destroy)

        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # create help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self._about)

        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menu_bar)

        self.deiconify()
    
    def _load_classifier(self):
        model_path = filedialog.askopenfilename()
        if model_path == "":
            messagebox.showerror("Error", "No classifier model selected.")
            return

        self._classifier_model = torch.load(model_path)
        self._classifier_model.to("cpu")

        self._classifier_model.eval()

        # show status that model is loaded
        self.status_bar['text'] = "Classifier model loaded."
        self.status_bar['fg'] = "green"
    
    def _sample_motion(self, motion):
        # sort motion from 2d blocks into a series with (row, column) coordinates
        motion_samples = []
        for r in range(0, motion.shape[0]):
            for c in range(0, motion.shape[1]):
                if motion[r, c] > 0:
                    motion_samples.append((r, c))
        
        return motion_samples

    def _run_tracking(self):
        if self.track_bar.processing_start is None or self.track_bar.processing_end is None:
            messagebox.showerror("Error", "Please set the processing start and end frames.")
            return

        # Create a new window
        task_window = tk.Toplevel(self)
        task_window.title("Tracking")
        task_window.geometry("300x100")

        # Create a progress bar
        progress = ttk.Progressbar(task_window, length=200)
        progress.pack(pady=20)

        # Create a cancel button
        cancel_button = tk.Button(task_window, text="Cancel", command=task_window.destroy)
        cancel_button.pack()

        # Start the task (update the progress bar)
        progress['value'] = 0
        track_start_frame = self.track_bar.processing_start
        track_end_frame = self.track_bar.processing_end

        progress_increment = 100.0 / (track_end_frame - track_start_frame)
        progress_value = 0.0

        for i in range(track_start_frame, track_end_frame):
            if i == 0 or i == self._num_frames:
                # we need the previous and the next frame to perform tracking, so just skip the first and last frame
                continue

            time.sleep(0.001)

            progress_value = progress_value + progress_increment
            progress['value'] = progress_value

            # perform tracking
            motion = self._motion_detector.detect(self._read_frame(i - 1), self._read_frame(i))

            positives = []

            # sample motion and test classifier
            for (i, j) in self._sample_motion(motion):
                row = i * self._motion_detector.block_size + self._motion_detector.block_size // 2
                col = j * self._motion_detector.block_size + self._motion_detector.block_size // 2

                # sample 32x32 and test classifier
                sample = self._current_frame[row - 16:row + 16, col - 16:col + 16]

                # Apply transformations to the image
                image_transformed = self._transform(sample)
                image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension

                # Pass the transformed image to the model
                output = self._classifier_model(image_transformed)

                if output > self._classification_threshold:
                    positives.append((output, (row, col)))
            
            if positives:
                # sort by output
                positives.sort(key=lambda x: x[0], reverse=True)

                # take the first one
                (_, coordinates) = positives[0]

                self._fish_tracking[i] = coordinates

            self.track_bar.mark_processed(i)

            self.update()

        # This makes the new window modal
        task_window.transient(self)
        task_window.grab_set()
        task_window.destroy()

        self.wait_window(task_window)
    
    def _set_track_start(self):
        self.track_bar.mark_processing_start(self.slider.get())

    def _set_track_end(self):
        self.track_bar.mark_processing_end(self.slider.get())

    def _motion_threshold_changed(self, value):
        self._motion_detector.threshold = float(value)

    def _reset_tracking(self):
        self.track_bar.clear()
    
    def _clear_tracking_frame(self):
        self.track_bar.mark_not_processed(self.slider.get())
    
    def _about(self):
        messagebox.showinfo("About", "Maze Tracking\n\nAuthor: Relja Ljubobratovic\nEmail: ljubobratovic.relja@gmail.com")

    def _save_tracking(self):
        pass
    
    def _load_video(self):
        filename = filedialog.askopenfilename()
        if filename == "":
            messagebox.showerror("Error", "No video selected.")
            return

        try:
            self.clip = VideoFileClip(filename)
        except Exception as e:
            messagebox.showerror("Error", "Failed to load video: {}".format(e))
            self.clip = None

        if self.clip is not None:
            self._calibrated = False
            self._calibration_running = False
            self._playing = False
            self._current_frame = None
            self._screen_points = []
            self._pixel_to_cm_ratio = 1.0

            # reset slider
            slider_length = self.clip.duration * self.clip.fps

            self.slider.destroy()
            self.slider = tk.Scale(self.bottom_frame, from_=0, to=slider_length, length=self._image_width, orient=tk.HORIZONTAL, command=self.slider_value_changed)
            self.slider.pack()

            self.slider.set(0)

            self._load_first_frame()

    def _calibrate(self):
        if self._calibrated:
            # warning popup to confirm calibration reset
            if not messagebox.askokcancel("Calibration Reset", "Are you sure you want to reset calibration?"):
                return
        elif self._calibration_running:
            # warning popup to confirm calibration cancel
            if messagebox.askokcancel("Calibration Cancel", "Are you sure you want to cancel calibration?"):
                # reset the text
                self.status_bar['text'] = "Calibration cancelled."
                self.status_bar['fg'] = "red"

                self._screen_points = []
                self.calibration_button['text'] = "Calibrate"
                self.calibration_button['fg'] = "red"

                # unbind click event
                self.image_label.unbind("<Button-1>")

                self._calibration_running = False
                self._calibrated = False

                # reset frame drawing to clear current points
                self.update_gui()
                return
        
        self._calibrated = False
        self._calibration_running = True

        # reset screen points
        self._screen_points = []

        # redraw frame to clear potentially preset calibration
        self.update_gui()

        self.status_bar['text'] = "Calibration started - left click to select corner points of the maze."
        self.status_bar['fg'] = "orange"

        self.image_label.bind("<Button-1>", self._calibration_click)

        # change button text
        self.calibration_button['text'] = "Cancel Calibration"
        self.calibration_button['fg'] = "purple"
    
    def _do_calibration(self):
        try:
            self._pixel_to_cm_ratio = self._maze_calibration.calibrate(self._screen_points)

            self._calibrated = True
            self._calibration_running = False

            # update status bar
            self.status_bar['text'] = "Calibration complete: pixel to cm ratio is {}".format(self._pixel_to_cm_ratio)
            self.status_bar['fg'] = "green"

            self.calibration_button['text'] = "Reset Calibration"
            self.calibration_button['fg'] = "green"

            self.run_tracking_button["state"] = tk.NORMAL
        except Exception as e:
            self._calibrated = False
            self._calibration_running = False

            self.status_bar['text'] = "Calibration failed - internal error: {}".format(e)
            self.status_bar['fg'] = "red"

            self.calibration_button['text'] = "Calibrate"
            self.calibration_button['fg'] = "red"

        # clear screen points
        self._screen_points = []

        # unbind click event
        self.image_label.unbind("<Button-1>")

        # reload current frame to rectify it
        self._reload_frame()

        # update gui to account for calibration presence
        self.update_gui()
    
    def _calibration_click(self, event):
    
        self._screen_points.append((event.x, event.y))

        self.update_gui()

        if len(self._screen_points) == 4:
            self._do_calibration()
            return

        # else update status bar
        self.status_bar['text'] = "Calibration point {} selected.".format(len(self._screen_points))
    
    def _get_current_frame_number(self):
        return self.slider.get()
    
    def _start_tracking(self):
        if self._playing:
            self._playing = False
            return

        self._playing = True
        threading.Thread(target=self._read_frames).start()
    
    def _read_frame(self, frame_number):
        frame = self.clip.get_frame(float(frame_number) / float(self.clip.fps))
        if self._calibrated:
            frame = self._maze_calibration.rectify_image(frame)
        return frame

    def _read_current_frame(self):
        return self._read_frame(self._get_current_frame_number())
    
    def _read_previous_frame(self):
        return self._read_frame(self._get_current_frame_number() - 1)

    def _pause_tracking(self):
        self._playing = False

    def _reload_frame(self):
        self._current_frame = self._read_current_frame()

        self._display_image = self._current_frame.copy()
    
    def _set_frame(self, frame):
        self.slider.set(frame)

    def _prev_frame(self):
        self._play_frame(self.slider.get() - 1)

    def _next_frame(self):
        self._play_frame(self.slider.get() + 1)
    
    def _track_frame(self):
        prevous_frame = self._read_previous_frame()
        motion = self._motion_detector.detect(prevous_frame, self._current_frame)
        self._draw_motion(motion)
    
    def _play_frame(self, frame):
        self._set_frame(frame)
        self._reload_frame()

        if self._calibrated:
            self._track_frame()
            self.track_bar.mark_processed(frame)

        self.update_gui()

    def _read_frames(self):
        while self._playing:
            time.sleep(0.5 / self.clip.fps)
            self.after(1, self._next_frame)

    def update_gui(self):
        self.update_image(self._display_image)
        self.label['text'] = self._frame_to_time(float(self._get_current_frame_number()), self.clip.fps)
    
    def _draw_motion(self, motion):
        motion_overlay = cv2.resize(motion, (self._display_image.shape[1], self._display_image.shape[0]))
        motion_overlay = cv2.cvtColor(motion_overlay, cv2.COLOR_GRAY2BGR)

        # scale to 255
        motion_overlay = (motion_overlay * 255).astype(np.uint8)
        
        # overlay with 50% transparency
        self._display_image = cv2.addWeighted(self._display_image, 0.5, motion_overlay, 0.5, 0)

    def update_image(self, image):
        if image is None:
            return

        image = image.copy()

        # draw screen points
        for point in self._screen_points:
            image = cv2.circle(image, point, 5, (0, 0, 255), -1)

        # Convert the image from OpenCV to PIL format
        image = Image.fromarray(image)

        # Convert the image to a format Tkinter can use
        image = ImageTk.PhotoImage(image)

        # Update the image_label to show the new image
        self.image_label.configure(image=image)

        # This line is necessary, because Tkinter tries to garbage collect the images
        self.image_label.image = image
    
    def slider_value_changed(self, value):
        self._current_frame = self._read_frame(int(value))
        self._display_image = self._current_frame.copy()
        self.update_gui()

    
    def _load_first_frame(self):
        self._current_frame = self._read_frame(0)
        self._display_image = self._current_frame.copy()

        assert self._current_frame is not None, "Failed to load first frame."

        self._image_width = self._current_frame.shape[1]

    def _frame_to_time(self, frame, fps):
        # Calculate the total seconds
        total_seconds = frame / fps

        # Calculate hours, minutes and seconds
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        # Format the time as hh:mm:ss
        time_str = "{:02}:{:02}:{:02}".format(hours, minutes, seconds)

        return time_str 
    


window = MainWindow()
window.mainloop()
