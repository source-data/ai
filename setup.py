import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai",
    version="0.2.1",
    python_requires='>=3.6',
    author="Thomas Lemberger",
    author_email="thomas.lemberger@embo.org",
    description="An AI toolbox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/source-data/ai",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "pillow"
    ],
    # keywords="",
    classifiers=(
        # full list: https://pypi.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries",
    ),
)
