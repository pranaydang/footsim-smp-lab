from setuptools import setup

setup(name='footsim',
      version='0.1',
      description='FootSim: Simulating tactile signals from the foot',
      url='https://github.com/ActiveTouchLab/footsim-python',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
      ],
      author='Katic N, Kazu Siqueira R, Cleland L, Strzalkowski N, Bent L, Raspopovic S, Saal HP',
      author_email='h.saal@sheffield.ac.uk',
      packages=['footsim'],
      install_requires=[
          'numpy','scipy','numba','matplotlib','holoviews','scikit-image'
      ],
      zip_safe=False,
      data_files=[('surfaces',['surfaces/foot.png'])])
