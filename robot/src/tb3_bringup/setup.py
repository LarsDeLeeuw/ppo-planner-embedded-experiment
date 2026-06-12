import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='Combined launch file for all TurtleBot3 research nodes',
    license='MIT',
    entry_points={},
)
