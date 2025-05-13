from setuptools import setup, find_packages

setup(
    name="dataglass",
    version="0.7.0",
    description="A Python library for data preprocessing, exploratory data analysis (EDA), and machine learning. "
                "This version implements the preprocessing module, including missing value handling, duplicate "
                "detection and handling, outlier handling, feature encoding, type conversion, feature scaling, and pipeline compatibility.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Saman Teymouri",
    author_email="saman.teymouri@gmail.com",
    license="BSD-3-Clause",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.21",
        "scikit-learn>=1.1",
        "matplotlib>=3.5",
        "rapidfuzz>=3.0",
        "seaborn>=0.11",
        "category_encoders>=2.5"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License"
    ],
    keywords=[
        "data-preprocessing", "eda", "machine-learning", "data-cleaning", 
        "feature-engineering", "pipeline", "pandas", "scikit-learn"
    ],
    project_urls={
        "Homepage": "https://github.com/samantim/dataglass",
        "Documentation": "https://github.com/samantim/dataglass/wiki",
        "Source": "https://github.com/samantim/dataglass",
        "Issues": "https://github.com/samantim/dataglass/issues"
    },
    include_package_data=True,
)
