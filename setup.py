from setuptools import setup, find_packages

setup(
    name="casino_holdem_ai",
    version="0.5.0",
    packages=find_packages('src'),
    py_modules=['cli'],
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "pandas",
        "eval7",
        "scikit-learn",
        "xgboost",
        "joblib",
        "tqdm"
    ],
    entry_points={
        'console_scripts': [
            'casino-ai=cli:main',
        ],
    },
)