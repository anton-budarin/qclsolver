from setuptools import setup, find_packages

setup(
     name='qclSolver',
     version='0.1',
     description='...',
     url='http://github.com/...',
     author='...',
     author_email='...',
     license='MIT',
     packages=find_packages(),
     python_requires='>=3.6',
     install_requires=['numpy', 
                     'scipy', 
                     'matplotlib',
                     'lxml'],
     zip_safe=False
     )
