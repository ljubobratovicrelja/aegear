import sys
import os

from setuptools import setup
import pyinstaller

# Add the PyInstaller hooks directory to the system path
sys.path.append(pyinstaller.__path__[0] + '/hooks')

# find absolute path of this module
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

# Define the setup parameters
setup(
    name='mazetracking',
    version='1.0',
    description='A module for tracking fish motion in a maze',
    author='Relja Ljubobratovic',
    author_email='ljubobratovic.relja@gmail.com',
    packages=['maze'],
    py_modules=['mazetracking'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'mazetracking = mazetracking:main'
        ]
    },
    options={
        'pyinstaller': {
            'hiddenimports': ['PIL', 'cv2', 'torch', 'torchvision', 'moviepy'],
            'console': True,
            'onefile': True,
            'name': 'mazetracking'
        }
    }
)