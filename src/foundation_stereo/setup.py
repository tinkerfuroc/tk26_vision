from setuptools import find_packages, setup

package_name = 'foundation_stereo'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/foundation_stereo.launch.py']),
        ('share/' + package_name + '/config',
            ['config/foundation_stereo.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cindy',
    maintainer_email='cindy.w0135@gmail.com',
    description='FoundationStereo + Fast-FoundationStereo ROS2 service/action + streaming depth node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'foundation_stereo_node = foundation_stereo.foundation_stereo_node:main',
        ],
    },
)
