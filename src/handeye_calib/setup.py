from setuptools import setup

package_name = 'handeye_calib'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tinker',
    maintainer_email='cindy.w0135@gmail.com',
    description='Eye-in-hand calibration for the wrist-mounted RealSense on the xArm.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'handeye_synthetic_check = handeye_calib.synthetic:main',
            'handeye_collect = handeye_calib.handeye_collect:main',
            'handeye_web = handeye_calib.handeye_web:main',
            'apply_handeye = handeye_calib.apply_handeye:main',
        ],
    },
)
