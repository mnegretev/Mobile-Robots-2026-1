from setuptools import setup
import os
from glob import glob

package_name = 'final_project_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # NO incluyas la línea de launch si no tienes archivos launch
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='melissa',
    maintainer_email='mmatiasz.410@gmail.com',
    description='Final project explorer for mobile robots',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_navigator = final_project_explorer.simple_navigator:main',
            'object_detector = final_project_explorer.object_detector:main',
        ],
    },
)