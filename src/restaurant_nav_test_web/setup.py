from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'restaurant_nav_test_web'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'webui'), glob('webui/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='Restaurant pure-nav test dashboard',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'restaurant_nav_test_web = restaurant_nav_test_web.restaurant_nav_test_web:main',
        ],
    },
)
