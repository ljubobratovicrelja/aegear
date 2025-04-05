#!/usr/bin/env python3
"""
Main entry point for the Aegear application.
This module sets up the GUI, handles video loading/calibration, and performs fish tracking.
"""

from aegear.gui.window import AegearMainWindow


if __name__ == '__main__':
    window = AegearMainWindow()
    window.mainloop()
