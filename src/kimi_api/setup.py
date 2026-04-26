from setuptools import find_packages, setup

package_name = 'kimi_api'

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
    description='LLM-backed feature extraction, matching, and grocery categorization (OpenRouter) for Tinker 2026',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feature_recognition = kimi_api.feature_recognition:main',
            'feature_matching = kimi_api.feature_matching:main',
            'grocery_categorize = kimi_api.grocery_categorize:main',
            'seat_recommend_bbox = kimi_api.seat_recommend_bbox:main',
        ],
    },
)
