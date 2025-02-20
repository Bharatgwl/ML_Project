from setuptools import find_packages, setup
from typing import List

HYPHEN_E = '-e .'
def get_requirements(path: str) -> List[str]:
    requirements = []
    with open(path, 'r') as f:
        for line in f:
            requirements = f.readlines()
            requirements=[req.replace("\n","") for req in requirements]
            
            if HYPHEN_E in requirements:
                requirements.remove(HYPHEN_E)
    return requirements
    #         requirements.append(line.strip())
    # return requirements
    

setup(
	name='mlproject',
	version='0.0.1',
	author='Bharat',
	author_email='bharatgusaiwal731@gmail.com',
	packages=find_packages(),
	install_requires=get_requirements('requirements.txt'),
)