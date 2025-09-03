import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/christopher/Mobile-Robots-2026-1/install/uf_ros_lib'
