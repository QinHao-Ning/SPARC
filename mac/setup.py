from pathlib import Path
import setuptools

package_root = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (package_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setuptools.setup(name='mac',
version='0.1',
description='Fast algebraic connectivity maximization',
url='https://github.com/MarineRoboticsGroup/mac',
author='keevindoherty',
install_requires=requirements,
extras_require={'cholesky': ['scikit-sparse']},
author_email='kdoherty@mit.edu',
packages=setuptools.find_packages(where=str(package_root)),
python_requires='>=3.8',
zip_safe=False)
