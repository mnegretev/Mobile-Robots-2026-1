from setuptools import find_packages, setup

package_name = 'proyecto'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axel',
    maintainer_email='theaxelruiz@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'image_listener = proyecto.image_listener:main',
        'yolo_node = proyecto.yolo_node:main',
        'random_goal_sender = proyecto.random_goal_sender:main',
        ],
    },
)
