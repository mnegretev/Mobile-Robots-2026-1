from setuptools import setup

package_name = 'object_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'std_msgs',
        'cv_bridge',
        'ultralytics',
        'opencv-python',
        'torch',
        'torchvision',
        'numpy<2'
    ],
    zip_safe=True,
    maintainer='catcyber02',
    maintainer_email='danixbalanquepz@gmail.com',
    description='Nodo de detección de objetos con YOLO',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = object_detector.detector_node:main',
        ],
    },
)
