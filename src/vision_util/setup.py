from setuptools import find_packages, setup

package_name = 'vision_util'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='Utility services (point-cloud relay, door detection) for Tinker 2026 vision',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'door_detection = vision_util.door_detection:main',
            'get_point_cloud = vision_util.get_point_cloud:main',
        ],
    },
)
