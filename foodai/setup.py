import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# this grabs the requirements from requirements.txt
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setuptools.setup(
    name='azumio_foodai',
    version='1.0.2',
    author='Igor Rendulic',
    author_email='igor@azumio.com',
    description='Azumio - Instant Food Recognition',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/azumio/foodai-clients/tree/master/foodai',
    project_urls={
        "Bug Tracker": "https://github.com/azumio/foodai-clients/issues"
    },
    license='Apache 2.0',
    packages=['azumio_foodai'],
    install_requires=REQUIREMENTS,
)