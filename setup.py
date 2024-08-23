from setuptools import setup, find_packages

# Lire les dépendances depuis requirements.txt
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="drug_smile",  # Remplacez par le nom de votre projet
    version="0.1.0",
    description="Predict New Medicines with AI",
    author="Lucas SEDRAN",
    url="https://github.com/lucas-sedran/drug_smile",  # Remplacez par l'URL de votre projet si applicable
    packages=find_packages(),  # Trouve automatiquement les packages dans votre projet
    install_requires=requirements,  # Installe les dépendances depuis requirements.txt
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
