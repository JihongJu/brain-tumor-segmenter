from setuptools import setup, find_packages


setup(name='brain-tumor-segmenter',
      version='1.0',
      author='Jihong Ju',
      author_email='daniel.jihong.ju@gmail.com',
      packages=find_packages(exclude=('tests', 'output')),
      #install_requires=["h5py",
      #                  "numpy",
      #                  "scipy",
      #                  "pandas",
      #                  "imbalanced-learn",
      #                  "scikit-learn",
      #                  "SimpleITK",
      #                  "medpy",
      #                  "requests",
      #                  "pytest"]
     )
