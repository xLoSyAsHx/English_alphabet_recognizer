from setuptools import setup
import alphabet_recogniser

with open('README.md', 'r') as f:
   LONG_DESCRIPTION = f.read()

CLASSIFIERS = [
    'Development Status :: 1 - Planning',
    'Intended Audience :: Developers',
    'License :: Free for non-commercial use',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.7',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

setup(
    name='alphabet_recogniser',
    version=alphabet_recogniser.__version__,
    author="xLoSyAsHx",
    author_email="https://github.com/xLoSyAsHx",
    url='https://github.com/xLoSyAsHx/HSE_ML_alphabet_recognition',
    description='Alphabet recognition model and utilities',
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    keywords="alphabet recognition example examples",
    test_suite='tests',
    python_requires='>=3.7.*',
    install_requires=[
        'torch>=1.4.0',
        'torchvision>=0.5.0',
        'Pillow',
        'scikit-image',
        'compress-pickle>=1.1.1'
    ],
    platforms=['Windows 10'],
    license='NO'
)
