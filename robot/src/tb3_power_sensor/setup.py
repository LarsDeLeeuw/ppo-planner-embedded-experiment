import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_power_sensor'

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
    install_requires=[
        'setuptools',
        'smbus2',
    ],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='INA219 power-sensing node + SBC thermal telemetry for the TurtleBot3 power experiment',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'power_sensor_node = tb3_power_sensor.power_sensor_node:main',
        ],
    },
)
