import os
from setuptools import setup
import shutil


if __name__ == '__main__':
    # Hypothetically, somebody might run this from somewhere other than the
    # root directory.
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Read in requirements.txt
    req_path = os.path.join(root_path, 'requirements.txt')
    with open(req_path, 'r') as reqs_file:
        requirements = list(reqs_file.readlines())

    setup(
        name='video2vision',
        version='0.1',
        author='Hanley Color Lab',
        author_email='dhanley@gmu.edu',
        url='https://github.com/HanleyColorLab/video2vision',
        packages=['video2vision'],
        setup_requires=requirements,
        install_requires=requirements,
    )
