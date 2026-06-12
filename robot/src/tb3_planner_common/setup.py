from setuptools import find_packages, setup

package_name = 'tb3_planner_common'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.24',
    ],
    zip_safe=True,
    maintainer='LarsDeLeeuw',
    maintainer_email='deleeuwlars@icloud.com',
    description='Planner-neutral Python utilities shared by grid-navigation planner packages',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [],
    },
)
