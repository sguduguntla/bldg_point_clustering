from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from os import path

reqs = [r.name for r in parse_requirements(open('requirements.txt'))]

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name="bldg_point_clustering",
    version="0.0.1",
    author="Sriharsha Guduguntla",
    author_email="sguduguntla@berkeley.edu",
    description="A Python 3.5+ wrapper for clustering building point labels using KMeans, DBScan, and Agglomerative clustering",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=[('.', ['bldg_point_clustering/helper/heuristics.yml'])],
    install_requires = reqs,
    python_requires='>=3.5',
)