import sys

# get path to this module
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

# add parent directory to path
sys.path.append(os.path.dirname(this_dir))

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

from maze.mazecalibration import MazeCalibration
from maze.motiondetection import MotionDetector

from maze.trajectory import trajectoryLength, smoothTrajectory, drawTrajectory
from maze.utils import ToolTip

# needed for the classifier loading
from maze.classifier import Classifier


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
    
    def unmark_processing_start(self):
        if self.processing_start is not None:
            self.delete("processing_start")
            self.processing_start = None
    
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
    
    def unmark_processing_end(self):
        if self.processing_end is not None:
            self.delete("processing_end")
            self.processing_end = None

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

    ONE_LINE_HEIGHT = 16

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
        self._first_frame_position = None

        # drawing variable
        self._draw_trajectory = tk.BooleanVar()
        self._draw_trajectory.set(True)

        self._save_tracking_check = tk.BooleanVar()
        self._save_tracking_check.set(True)

        self._tracking_crosscheck = tk.BooleanVar()
        self._tracking_crosscheck.set(True)


        self._fish_tracking = {}
        self._trajectory_smooth_size = 9
        self._trajectory_frame_skip = 3

        self._classification_threshold = 0.99  # we can expect classifier to be very confident
        self._classifier_model = None

        # classifier transformations 
        self._transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert np array to PIL Image
            transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
        ])

        # screen points for calibration and other purposes
        self._screen_points = []

        # initiate maze calibration utility
        self._maze_calibration = MazeCalibration("data/calibration.xml")

        # motion detector
        self._motion_detector = MotionDetector(10, 3, 15, 800, 2500)

        self.dialog_window = tk.Toplevel(self)
        self.dialog_window.withdraw()

        self.dialog_window.title("Load Video")

        #### DEBUG PART ######
        initial_video = filedialog.askopenfilename(parent=self.dialog_window)
        #initial_video = "data/videos/K9.MOV"
        #initial_video = "data/videos/EE1.MOV"

        self._classifier_model = torch.load("data/models/model_cnn4_v3.pth")
        self._classifier_model.to("cpu")

        self._classifier_model.eval()
        #### DEBUG PART END ######

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

        # Create a list of videos on the left side to the image_label
        #self.video_listbox = tk.Listbox(self.center_frame, width=20, height=10)
        #self.video_listbox.grid(row=0, column=0, sticky="nsew")

        # Create a label to hold the image
        self.image_label = tk.Label(self.center_frame, cursor="cross")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # scrollbar for the listbox
        self.scrollbar = tk.Scrollbar(self.center_frame, orient=tk.VERTICAL)
        self.scrollbar.grid(row=0, column=2, sticky="nsew")

        # Create a listbox for all tracked frames
        self.tracking_listbox = tk.Listbox(self.center_frame, width=30, height=10, yscrollcommand=self.scrollbar.set)
        self.tracking_listbox.grid(row=0, column=1, sticky="nsew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # add item select callback
        self.tracking_listbox.bind('<<ListboxSelect>>', self._listbox_item_selected)

        # add del key callback
        self.tracking_listbox.bind('<Delete>', self._listbox_item_deleted)

        # configure the scrollbar
        self.scrollbar.config(command=self.tracking_listbox.yview)

        self._load_first_frame()

        self.slider = tk.Scale(self.bottom_frame, from_=0, to=self._num_frames, length=self._image_width, orient=tk.HORIZONTAL, command=self.slider_value_changed)
        self.slider.pack()

        self.track_bar = TrackingBar(self.bottom_frame, self._num_frames, width=self._image_width, height=10)
        self.track_bar.pack()

        self.label = tk.Label(self.bottom_frame, text="00:00:00")
        self.label.pack()

        self.calibration_button = tk.Button(self.right_frame, text="Calibrate", command=self._calibrate, fg="red")
        self.calibration_button.pack(side=tk.LEFT)

        self.set_track_start_button = tk.Button(self.right_frame, text="Set Track Start", command=self._set_track_start, state=tk.DISABLED)
        self.set_track_start_button.pack(side=tk.LEFT)

        self.set_track_end_button = tk.Button(self.right_frame, text="Set Track End", command=self._set_track_end, state=tk.DISABLED)
        self.set_track_end_button.pack(side=tk.LEFT)

        self.run_tracking_button = tk.Button(self.right_frame, text="Run Tracking", command=self._run_tracking, state=tk.DISABLED)
        self.run_tracking_button.pack(side=tk.LEFT)

        # Checkbutton to save tracking automatically
        #self.save_tracking_checkbox = tk.Checkbutton(self.right_frame, text="Save Tracking", variable=self._save_tracking_check)
        #self.save_tracking_checkbox.pack(side=tk.LEFT)

        # checkbox for tracking crosschecking
        self.crosscheck_checkbox = tk.Checkbutton(self.right_frame, text="Crosscheck", variable=self._tracking_crosscheck)
        self.crosscheck_checkbox.pack(side=tk.LEFT)

        # set tooltip for crosscheck checkbox
        self.chrosscheck_tooltip = ToolTip(self.crosscheck_checkbox, "This runs classification on mutliple consecutive frames to ensure that the fish is correctly identified.")
        self.crosscheck_checkbox.bind("<Enter>", self.chrosscheck_tooltip.display_tooltip)
        self.crosscheck_checkbox.bind("<Leave>", self.chrosscheck_tooltip.hide_tooltip)

        self.reset_tracking_button = tk.Button(self.right_frame, text="Reset Tracking", command=self._reset_tracking)
        self.reset_tracking_button.pack(side=tk.LEFT)

        # scale for classification threshold
        self.classification_threshold_scale = tk.Scale(self.right_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Classification Threshold", command=self._classification_threshold_changed)
        self.classification_threshold_scale.set(self._classification_threshold * 100)
        self.classification_threshold_scale.pack(side=tk.LEFT)

        # scale for trajectory smoothing
        self.smooth_trajectory_scale = tk.Scale(self.right_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="Trajectory Smoothing", command=self._trajectory_smooth_size_changed)
        self.smooth_trajectory_scale.set(self._trajectory_smooth_size)
        self.smooth_trajectory_scale.pack(side=tk.LEFT)

        # scale for tracking every nth frame
        self.tracking_frame_scale = tk.Scale(self.right_frame, from_=1, to=100, orient=tk.HORIZONTAL, label="Tracking Frame Skip", command=self._trajectory_frame_skip_changed)
        self.tracking_frame_scale.set(self._trajectory_frame_skip)
        self.tracking_frame_scale.pack(side=tk.LEFT)

        self.draw_trajectory_checkbox = tk.Checkbutton(self.right_frame, text="Draw Trajectory", variable=self._draw_trajectory)
        self.draw_trajectory_checkbox.pack(side=tk.LEFT)

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

        # status bar specifically dedicated toward recording the travel distance
        self.distance_status_bar = tk.Label(self.bottom_frame, text="Distance: 0.0 cm", fg="blue", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.distance_status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_gui()

        # create menu bar
        self.menu_bar = tk.Menu(self)

        # create file menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load Video", command=self._load_video)
        self.file_menu.add_command(label="Load Classifier Model", command=self._load_classifier)
        self.file_menu.add_command(label="Load Tracking", command=self._load_tracking)
        self.file_menu.add_command(label="Save Tracking", command=self._save_tracking)
        self.file_menu.add_command(label="Exit", command=self.destroy)

        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # create help menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self._about)

        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menu_bar)

        self.deiconify()
    
    def _listbox_item_selected(self, event):
        sel = self.tracking_listbox.curselection()
        if not sel:
            return

        item_selected = sel[0]

        # get list item
        frame_id = int(self.tracking_listbox.get(item_selected).split(":")[0])

        self._set_frame(frame_id)
        self.update_gui()
    
    def _listbox_item_deleted(self, event):
        item_selected = self.tracking_listbox.curselection()[0]

        # get list item
        frame_id = int(self.tracking_listbox.get(item_selected).split(":")[0])
        self.track_bar.mark_not_processed(frame_id)
        del(self._fish_tracking[frame_id])

        # remove item from listbox
        self.tracking_listbox.delete(item_selected)

        self.update_gui()
    
    def _classification_threshold_changed(self, value):
        self._classification_threshold = float(value) / 100.0
    
    def _trajectory_smooth_size_changed(self, value):
        v = int(value)
        # enforce odd number
        if v % 2 == 0:
            v += 1
        
        self._trajectory_smooth_size = v

        # set the value to gui
        self.smooth_trajectory_scale.set(v)

        self.update_gui()

    def _trajectory_frame_skip_changed(self, value):
        v = int(value)
        self._trajectory_frame_skip = v
        
    def _load_classifier(self):
        model_path = filedialog.askopenfilename()
        if model_path == "":
            messagebox.showerror("Error", "No classifier model selected.")
            return

        device = torch.device("cpu")

        self._classifier_model = Classifier()
        self._classifier_model.load_state_dict(torch.load(model_path, map_location=device))
        self._classifier_mode.to(device)

        self._classifier_model.eval()

        # show status that model is loaded
        self.status_bar['text'] = "Classifier model loaded."
        self.status_bar['fg'] = "green"
    
    def _run_tracking(self):
        if self._classifier_model is None:
            messagebox.showerror("Error", "Please load a classifier model first.")
            return
        
        if self.track_bar.processing_start is None or self.track_bar.processing_end is None:
            messagebox.showerror("Error", "Please set the processing start and end frames.")
            return

        # Create a new window
        task_window = tk.Toplevel(self)
        task_window.title("Tracking")
        task_window.geometry("300x120")

        # label showing the current frame
        progress_label = tk.Label(task_window, text="Progress: 0%")
        progress_label.pack()

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

        progress_increment = 100.0 / ((track_end_frame - track_start_frame) / self._trajectory_frame_skip)
        progress_value = 0.0

        # size of the classifiers window
        sample_size = 32
        sample_size_half = sample_size // 2

        # measure estimated time as well
        start_time = time.time()

        for frame_id in range(track_start_frame, track_end_frame, self._trajectory_frame_skip):
            # update the label

            if frame_id == 0 or frame_id == self._num_frames:
                # we need the previous and the next frame to perform tracking, so just skip the first and last frame
                continue

            progress_value = progress_value + progress_increment
            progress['value'] = progress_value

            # calculate the estimated time
            elapsed_time = time.time() - start_time
            estimated_time = elapsed_time / progress_value * 100.0 - elapsed_time

            # write estimated time in ss:mm:hh format
            estimated_time_str = "{:02d}:{:02d}:{:02d}".format(int(estimated_time // 3600), int((estimated_time // 60) % 60), int(estimated_time % 60))

            progress_label['text'] = "Progress: {}%, estimated time: {}".format(int(progress_value), estimated_time_str)

            # positive hits, potential fish locations
            positives = []

            prev_frame_image = self._read_frame(frame_id-1)
            frame_image = self._read_frame(frame_id)
            next_frame_image = self._read_frame(frame_id)

            draw_image = frame_image.copy()

            # turn to RGB
            prev_frame_image = cv2.cvtColor(prev_frame_image, cv2.COLOR_BGR2RGB)
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            next_frame_image = cv2.cvtColor(next_frame_image, cv2.COLOR_BGR2RGB)

            # perform tracking
            good_contours, bad_contours = self._motion_detector.detect2(prev_frame_image, frame_image, next_frame_image)

            # draw contours
            cv2.drawContours(draw_image, good_contours, -1, (0, 255, 0), 2)
            cv2.drawContours(draw_image, bad_contours, -1, (0, 0, 255), 2)

            for contour in good_contours:
                # find center of the contour
                M = cv2.moments(contour)
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                # draw center
                draw_image = cv2.circle(draw_image, (x, y), 5, (0, 255, 0), -1)

                # given sample size, check if ROI is within the frame
                if x - sample_size_half < 0 or x + sample_size_half >= frame_image.shape[1] or y - sample_size_half < 0 or y + sample_size_half >= frame_image.shape[0]:
                    continue

                # Pass the transformed image to the model
                output = None
                if not self._tracking_crosscheck.get():
                    # sample 32x32 and test classifier
                    sample = frame_image[y - sample_size_half:y + sample_size_half, x - sample_size_half:x + sample_size_half]

                    # Apply transformations to the image
                    image_transformed = self._transform(sample)
                    image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension
                    output = self._classifier_model(image_transformed)
                else:
                    outputs = []
                    for image in [prev_frame_image, frame_image, next_frame_image]:
                        # sample 32x32 and test classifier
                        sample = image[y - sample_size_half:y + sample_size_half, x - sample_size_half:x + sample_size_half]

                        # Apply transformations to the image
                        image_transformed = self._transform(sample)
                        image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension
                        output = self._classifier_model(image_transformed)
                        #output_sum = output_sum + output
                        outputs.append(float(output))

                    # take the worst output
                    output = min(outputs)

                # print output weight
                if output > self._classification_threshold:
                    positives.append((output, contour, (x, y)))
            
            track_hit = None
            if positives:
                # sort by output
                positives.sort(key=lambda x: x[0], reverse=True)

                # take the first one
                (output, contour, coordinates) = positives[0]

                self._fish_tracking[frame_id] = (contour, coordinates)

                self.track_bar.mark_processed(frame_id)

                # draw coordinates on our draw_image
                cv2.circle(draw_image, coordinates, 5, (255, 0, 0), -1)

                # update the tracking_listbox - add this positive to the list
                self.tracking_listbox.insert(tk.END, "{}: {}".format(frame_id, float(output)))

            #cv2.imshow("debug", draw_image)
            #cv2.waitKey(1)

            # update the image
            self._display_image = draw_image.copy()
            self.update_image(self._display_image)



            self.update()

        # This makes the new window modal
        task_window.transient(self)
        task_window.grab_set()
        task_window.destroy()

        try:
            self.wait_window(task_window)
        except:
            pass

        # go to start frame
        self._play_frame(track_start_frame)
    
    def _set_track_start(self):
        self.track_bar.mark_processing_start(self.slider.get())

    def _set_track_end(self):
        self.track_bar.mark_processing_end(self.slider.get())
    
    def _reset_tracking(self):
        self.track_bar.clear()
        self._fish_tracking = {}

        # also reset start and end
        self.track_bar.unmark_processing_start()
        self.track_bar.unmark_processing_end()

        # update gui
        self.update_gui()
    
    def _about(self):
        messagebox.showinfo("About", "Maze Tracking\n\nAuthor: Relja Ljubobratovic\nEmail: ljubobratovic.relja@gmail.com")
    
    def _load_tracking(self):
        pass

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
        
        # draw tracking if present
        frame_id = self._get_current_frame_number()
        if frame_id in self._fish_tracking:
            (contour, (x, y)) = self._fish_tracking[frame_id]
            image = cv2.circle(image, (x, y), 12, (0, 255, 0))

            # draw contour
            image = cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        if self._draw_trajectory.get() and len(self._fish_tracking) > 1:
            # take trajectory up to this frame
            trajectory = []
            tracked_frames = sorted(list(self._fish_tracking.keys()))
            for frame_id in tracked_frames:
                if frame_id >= self._get_current_frame_number():
                    break

                (contour, ptn) = self._fish_tracking[frame_id]
                trajectory.append(ptn) 
            
            s_trajectory = smoothTrajectory(trajectory, self._trajectory_smooth_size)
            travelDistance = trajectoryLength(s_trajectory) * self._pixel_to_cm_ratio

            # update the travel distance status bar
            self.distance_status_bar['text'] = "Distance: {} cm".format(travelDistance)

            # draw the trajectory
            image = drawTrajectory(image, s_trajectory)
        
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

        # Create a temporary Text widget with one line and same font as your Listbox
        temp_text = tk.Text(self.tracking_listbox, height=1, font=("TkDefaultFont"))
        temp_text.pack()

        # Update the root to make sure widget sizes are calculated
        self.tracking_listbox.update()

        # Get the height of the text widget, which corresponds to the height of one line
        MainWindow.ONE_LINE_HEIGHT = temp_text.winfo_reqheight()

        # include the marging to another list item
        MainWindow.ONE_LINE_HEIGHT += 4

        temp_text.destroy()

        # set height of the tracking_listbox
        self.tracking_listbox.config(height=int(self._current_frame.shape[0] / MainWindow.ONE_LINE_HEIGHT))
        self.tracking_listbox.update()



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
