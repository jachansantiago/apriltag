import setuptools
from cmake_setuptools import CMakeExtension, CMakeBuildExt

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="apriltag",
    version="0.0.1",
    packages=setuptools.find_namespace_packages(include=['python']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=[CMakeExtension("apriltag")],
    cmdclass={'build_ext': CMakeBuildExt}
)
