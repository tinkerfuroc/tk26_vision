from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_track'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'webui'), glob('webui/*')),
        # Include model files
        ('share/' + package_name + '/models', glob('*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='Person tracking package using YOLO with re-identification',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_track_server = vision_track.person_track_node:main',
            'person_track_test_client = vision_track.person_track_test_client:main',
            'track_web = vision_track.track_web:main',
        ],
    },
)
