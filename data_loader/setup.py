from distutils.core import setup, Extension
import numpy as np

coviar_utils_module = Extension('coviar',
		sources = ['coviar_data_loader.c'],
		include_dirs=[np.get_include(), './ffmpeg/include/'],
		extra_compile_args=['-DNDEBUG', '-O3'],
		extra_link_args=['-lavutil', '-lavcodec', '-lavformat', '-lswscale', '-L./ffmpeg/lib/']
		#extra_link_args=['./ffmpeg/lib/avutil.lib', './ffmpeg/lib/avcodec.lib', './ffmpeg/lib/avformat.lib', './ffmpeg/lib/swscale.lib']
)

setup ( name = 'coviar',
	version = '0.1',
	description = 'Utils for coviar training.',
	ext_modules = [coviar_utils_module]
)
