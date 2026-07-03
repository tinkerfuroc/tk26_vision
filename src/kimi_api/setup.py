import os

from setuptools import find_packages, setup

package_name = 'kimi_api'


def _fewshot_data_files():
    """Walk fewshot/ and emit one (share-dest, [files]) per directory.

    ament_python's data_files preserves the relative tree, so each existing
    slug dir at build time becomes share/kimi_api/fewshot/<slug>/. New slug
    dirs added by the annotator after build require a rebuild to register
    (the glob runs at build time).
    """
    out = []
    if not os.path.isdir('fewshot'):
        return out
    for root, _, files in os.walk('fewshot'):
        files = [f for f in files if not f.startswith('.') or f == '.gitkeep']
        if not files:
            continue
        rel = os.path.relpath(root, '.')
        out.append((
            os.path.join('share', package_name, rel),
            [os.path.join(root, f) for f in files],
        ))
    return out


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *_fewshot_data_files(),
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
            'seat_fewshot_annotator = kimi_api.fewshot_annotator:main',
            'object_scan = kimi_api.object_scan:main',
        ],
    },
)
