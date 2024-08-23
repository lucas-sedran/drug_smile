from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='drug_smile',
      version="0.0.1",
      description="",
      license="",
      author="Lucas Sedran, Issam Mehnana, Benoit Cochet, Dorian Schnepp",
      author_email="contact@lewagon.org",
      url="https://github.com/lucas-sedran/drug_smile",
      url_interface = "https://github.com/DodooHellio/drug_smile_interface",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
