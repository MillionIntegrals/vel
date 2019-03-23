from setuptools import setup, find_packages

long_description = """
This repository is my project to bring velocity to deep-learning research, by providing tried and tested large pool of
prebuilt components that are known to be working well together.

I would like to minimize time to market of new projects, ease experimentation and provide bits of experiment management
to bring some order to the data science workflow.

Ideally, for most applications it should be enough to write a config file wiring existing components together.
If that's not the case writing bits of custom code shouldn't be unnecessarily complex.

This repository is still in an early stage of that journey but it will grow as I'll be putting some work into it.
"""


setup(
    name='vel',
    version='0.3.0',
    description="Velocity in deep-learning research",
    long_description=long_description,
    url='https://github.com/MillionIntegrals/vel',
    author='Jerry Tworek',
    author_email='jerry@millionintegrals.com',
    license='MIT',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires='>=3.6',
    install_requires=[
        'attrs',
        'cloudpickle',
        'numpy',
        'opencv-python',
        'pandas',
        'pyyaml',
        'scikit-learn',
        'torch ~= 1.0',
        'torchvision',
        'torchtext',
        'tqdm'
    ],
    extras_require={
        'visdom': ['visdom'],
        'mongo': ['pymongo', 'dnspython'],
        'gym': ['gym[atari,box2d,classic_control]'],
        'mujoco': ['gym[mujoco,robotics]'],
        'dev': ['pytest', 'ipython', 'jupyter'],
        'text': ['spacy'],
        'all': ['visdom', 'pymongo', 'dnspython', 'gym[all]', 'pytest', 'spacy', 'ipython', 'jupyter']
    },
    tests_require=[
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'vel = vel.launcher:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    scripts=[]
)
