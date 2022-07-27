from setuptools import setup


setup(name='spheroid-tracking',
      version="0.1",
      packages=['spheroid-tracking'],
      description='',
      author='johannes-bartl',
      author_email='johannes.bartl@fau.de',
      license='GPLv3',
      install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'tqdm',
            'tifffile',
            "opencv-python",
            "pandas",
            "trackpy",
            "shapely",
            "jupyter",
            "pims"
      ],
)
