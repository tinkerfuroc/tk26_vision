import os
from glob import glob

from setuptools import setup

package_name = 'handeye_calib'

setup(
    name=package_name,
    version='0.1.0',
    # Listing webui as a sub-"package" silences setuptools' "an importable
    # directory was found that wasn't declared" warning. webui has no .py
    # files, but ament_python treats every dir under the package root as a
    # potential package — declaring it is the friction-free path.
    packages=[package_name, package_name + '.webui'],
    # Ship the static webui assets next to handeye_web.py in site-packages,
    # so `Path(__file__).parent / "webui"` resolves from the install tree
    # (the FastAPI server uses that path for /static and FileResponse).
    package_data={package_name: ['webui/*']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Mirror the assets into share/ for parity with pan_tilt's layout
        # (a future operator who runs `ros2 pkg prefix handeye_calib` will
        # find them where they expect).
        (os.path.join('share', package_name, 'webui'),
         glob('handeye_calib/webui/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tinker',
    maintainer_email='cindy.w0135@gmail.com',
    description='Eye-in-hand calibration for the wrist-mounted RealSense on the xArm.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'handeye_synthetic_check = handeye_calib.synthetic:main',
            'handeye_collect = handeye_calib.handeye_collect:main',
            'handeye_web = handeye_calib.handeye_web:main',
            'apply_handeye = handeye_calib.apply_handeye:main',
        ],
    },
)
