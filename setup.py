import setuptools

from skbuild import setup


cmake_args = ["-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++","-DCMAKE_BUILD_TYPE=Release"]

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="apriltag",
    version="0.0.1",
    packages=["apriltag"],
    package_dir = {"":"python"},
    cmake_args=cmake_args,
    include_package_data=True,
    cmake_install_dir="python/apriltag",
    cmake_source_dir=".",
    cmake_with_sdist=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False,
    setup_requires=["cmake"]
    )
