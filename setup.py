import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lripy',
    version='0.0.1',
    author='Christian Grussler',
    author_email='christian.grussler@eng.cam.ac.uk',
    license='GPLv3',
    zip_safe=False,
    url='http://github.com/LowRankOpt/LRIPy',
    description='Python implementation for optimization with Low-Rank Inducing Norms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['lripy'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)