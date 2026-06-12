import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb3_diagnostics'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='Per-host ambient diagnostics for the TurtleBot3 power experiment',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'diagnostics_node = tb3_diagnostics.diagnostics_node:main',
        ],
    },
)
