import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phymlq",
    version="v0.1.1",
    author="Animesh Sinha, Jai Bardhan, Kalp Shah",
    author_email="animesh.sinha@research.iiit.ac.in, jai.bardhan@research.iiit.ac.in, kalp.shah@research.iiit.ac.in",
    description="Helper Methods for several paper replications related to machine learning and quantum computation, physics oriented in particular.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jai2500/particle-tagging/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)