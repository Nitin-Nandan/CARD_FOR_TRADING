from setuptools import setup, find_packages

setup(
    name="pandas_ta",
    version="0.3.14b0",
    description="An easy to use Technical Analysis library with over 130 Indicators and Utility functions.",
    author="Kevin Johnson",
    packages=find_packages(),
    install_package_data=True,
    install_requires=["pandas"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
