from pathlib import Path
from setuptools import setup, find_packages

def read_requirements(filename="requirements.txt"):
    req_path = Path(__file__).parent / filename
    if not req_path.exists():
        return []
    return [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="stroke-gait-analysis",
    version="0.1.0",
    description="Stroke gait analysis using RQA, classical ML, and CNN pipelines",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=["paths", "config"],
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
