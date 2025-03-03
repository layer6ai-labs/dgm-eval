import os

import setuptools


def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()


if __name__ == '__main__':
    setuptools.setup(
        name='dgm-eval',
        author='Layer 6',
        description=('Package for evaluating deep generative models'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        packages=['dgm_eval'],
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
        ],
        python_requires='>=3.7',
        entry_points={
            'console_scripts': [
                'dgm-eval = dgm_eval:main',
            ],
        },
        install_requires=[
            'numpy==1.23.3',
            'opencv-python==4.6.0.66',
            'open_clip_torch==2.19.0',
            'pandas==1.5.3',
            'pillow==9.2.0',
            'scikit-image==0.19.3',
            'scikit-learn==1.1.3',
            'scipy==1.9.3',
            'timm==0.8.19.dev0',
            'torch>=2.0.0',
            'torchvision>=0.2.2',
            'transformers==4.26.0',
            'xformers>=0.0.18',
        ],
        extras_require={'dev': ['flake8',
                                'flake8-bugbear',
                                'flake8-isort',
                                'nox']},
    )
