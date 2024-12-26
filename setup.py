import setuptools
import src

setuptools.setup(
    version=__version__,
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["chimpstackr=run:main"]},
)