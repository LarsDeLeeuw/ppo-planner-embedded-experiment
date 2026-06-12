import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ppo_planner'

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
        (os.path.join('share', package_name, 'models'),
            glob('models/*.onnx')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.21',
        'onnxruntime>=1.17,<2.0',
    ],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='ROS2 node wrapping an ONNX-exported PPO policy (numpy + onnxruntime, no torch) for grid navigation planning',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ppo_planner_node = ppo_planner.ppo_planner_node:main',
        ],
    },
)
