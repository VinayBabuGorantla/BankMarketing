from setuptools import find_packages, setup
from typing import List

# Constant for '-e .' in requirements
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads the requirements file and returns a list of requirements.
    It also removes the '-e .' if present.
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Replacing newlines and stripping extra spaces

        # Remove '-e .' if present, which is for editable installation
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="Bank Marketing - Product Subscription Prediction",  # Project name
    version="0.0.1",  # Initial version
    author="Vinay Gorantla",  # Your name
    author_email="vinayc.gorantla@gmail.com",  # Your contact email
    description="A model to predict product subscriptions based on bank marketing data.",  # Optional project description
    packages=find_packages(),  # Automatically find packages in the project
    install_requires=get_requirements("requirements.txt"),  # Pulls dependencies from requirements.txt
    license="MIT",  # Optional license, change if necessary
)
