from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
VERSION = '2.3.0'
# Setting up
setup(
    name="ExtremeLy",
    version=VERSION,
    description="Extrem value Analysis in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Surya Lamichaney",
    author_email="suryalamichaney38741@gmail.com",
    url="https://github.com/SURYA-LAMICHANEY/ExtremeLy",
    packages=["ExtremeLy"],
    install_requires=["evt",],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license="MIT",
    keywords=["Extreme Value Analysis",
              "Block Maxima",
              "Peak-Over-Threshold",
              "Generalized Extreme Value Distribution (GEV)",
              "Generalized Pareto Distribution (GPD)"],
    platforms=["linux", "windows", "mac"],
    python_requires=">=3.8",
    project_urls={
        "GitHub": "https://github.com/SURYA-LAMICHANEY/ExtremeLy",
        "PyPI": "https://pypi.org/project/ExtremeLy/",
        "Documentation": "https://surya-lamichaney.github.io/ExtremeLy/",
    },
)
