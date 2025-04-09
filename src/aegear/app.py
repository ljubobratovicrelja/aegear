#!/usr/bin/env python3
"""
Main entry point for the Aegear application.
This module sets up the GUI, handles video loading/calibration, and performs fish tracking.
"""

from aegear.gui.window import AegearMainWindow

def main():
    """
    Main function to run the Aegear application.
    Initializes the main window and starts the GUI event loop.
    """
    window = AegearMainWindow()
    window.mainloop()

if __name__ == '__main__':
    main()
