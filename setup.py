import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="magiccluster",
    version="0.0.3",
    author="junhao.wen",
    author_email="junhao.wen89@email.com",
    description="Multi-scale semi-supervised clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbai106/MAGIC",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'magiccluster = magiccluster.main:main',
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
