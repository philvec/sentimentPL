import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentimentpl",
    version="0.0.5",
    author="Filip Strzałka",
    author_email="strzalkafilip@gmail.com",
    description="PyTorch models for polish language sentiment regression based on allegro/herbert and CLARIN-PL dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/philvec/sentimentPL",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['torch>=1.7.1', 'transformers', 'tqdm', 'matplotlib']
)
