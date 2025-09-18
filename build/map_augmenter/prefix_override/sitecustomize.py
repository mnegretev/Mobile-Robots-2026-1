import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/melissa/Documentos/Robots_Mobiles/Mobile-Robots-2026-1/install/map_augmenter'
