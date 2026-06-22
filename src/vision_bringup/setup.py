import os
from glob import glob

from setuptools import setup

package_name = 'vision_bringup'

# Single source of truth: the FastDDS SHM profile + RealSense QoS overrides live
# at src/tk26_vision/config/ (this package is at src/tk26_vision/src/<pkg>/).
# Install copies into the package share so the launch files can resolve them via
# FindPackageShare. colcon requires data_files sources to be RELATIVE to the
# package dir, so reference them as ../../config/* (not an absolute path).
_tk26_config = os.path.join('..', '..', 'config')

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), [
            os.path.join(_tk26_config, 'fastdds_shm.xml'),
            os.path.join(_tk26_config, 'realsense_qos.yaml'),
        ]),  # noqa: E501 -- relative ../../config sources, see comment above
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ccindy0171',
    maintainer_email='cindy.w0135@gmail.com',
    description='Composed bringup launch files for the tk26 vision stack.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
