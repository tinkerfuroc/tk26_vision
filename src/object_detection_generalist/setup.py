from setuptools import find_packages, setup

package_name = 'object_detection_generalist'

setup(
    name=package_name,
    version='0.1.0',
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
    description='Clean YOLO + optional VLM (Gemini 2.5 Pro) bbox + FastSAM '
                'mask generalist object detection service.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'generalist_node = object_detection_generalist.generalist_node:main',
        ],
    },
)
