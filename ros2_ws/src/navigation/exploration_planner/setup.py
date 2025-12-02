from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'exploration_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # <<<<<< INSTALAR LAUNCH >>>>>>>
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='catcyber02',
    maintainer_email='danixbalanquepz@gmail.com',
    description='Exploration planner',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # <<<<<< TUS EJECUTABLES >>>>>>>
            'explorer = exploration_planner.explorer:main',
            'cell_marker = exploration_planner.cell_marker:main',
            'Data_Yolo = exploration_planner.Data_Yolo:main'
        ],
    },
)
