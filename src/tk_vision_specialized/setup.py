from glob import glob

from setuptools import find_packages, setup

package_name = 'tk_vision_specialized'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        # Reference items map for object_match_server. Lives at
        # src/tk26_vision/src/items/ (sibling of this package); installed
        # into share/ so ament_index can resolve it after colcon build.
        ('share/' + package_name + '/items',
            glob('../items/items_map.yaml') + glob('../items/*.jpg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='Specialized vision nodes for shelf object detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spot_on_shelf_server = tk_vision_specialized.spot_on_shelf_server:main',
            'waving_person_server = tk_vision_specialized.waving_person_server:main',
            'waving_client = tk_vision_specialized.waving_client:main',
            'waving_bench = tk_vision_specialized.waving_bench:main',
            'check_waving_inference = tk_vision_specialized.check_waving_inference:main',
            'placing_location_server = tk_vision_specialized.placing_location_server:main',
            'object_match_server = tk_vision_specialized.object_match_server:main',
            'object_match_all_server = tk_vision_specialized.object_match_all_server:main',
        ],
    },
)
