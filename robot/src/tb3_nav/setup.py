import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='Closed-loop grid navigation node for TurtleBot3',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'grid_nav_node = tb3_nav.grid_nav_node:main',
            'calibrate = tb3_nav.calibrate:main',
            'calibrate_hop = tb3_nav.calibrate_hop_scale:main',
        ],
    },
)
