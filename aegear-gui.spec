# aegear-gui.spec

import sys

from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
from pathlib import Path

project_root = Path.cwd()

# Include all submodules of aegear
hidden_imports = collect_submodules("aegear")

# Collect files from the 'data' folder
data_folder = project_root / "data"
data_files = [
    ("media/icon.ico", "media"),
    ("media/logo.png", "media"),
    ("data/calibration.xml", "data"),
    ("data/models/model_cnn3_2023-08-16.pth", "data/models"),
    ("data/models/model_efficient_unet_2025-04-04.pth", "data/models"),
    ("data/models/model_siamese_2025-04-23.pth", "data/models"),
]

block_cipher = None

a = Analysis(
    ['src/aegear/app.py'],  # main script path
    pathex=['src'],  # so aegear is found in src/
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == 'darwin':
    icon_file = 'media/icon.icns'
else:
    icon_file = 'media/icon.ico'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='aegear-gui',
    debug=False,
    strip=False,
    upx=True,
    console=False,  # set to True if you want a console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file
)
