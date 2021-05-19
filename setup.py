from setuptools import setup
 
with open('README.md') as file: 
    long_description = file.read()
 
setup(
    name='ddsp',
    description='Code for DDSP_PyTorch',
    version='0.0.1',
    author='',
    author_email='',
    install_requires=[
        'numpy>=1.19.4',
        'crepe>=0.0.11',
        'librosa>=0.8.0',
        'einops>=0.3.0',
        'tqdm>=4.46.0',
        'torch>=1.7.0',
        'effortless_config>=0.7.0',
        'SoundFile>=0.10.3.post1',
        'Flask>=1.1.2',
        'PyYAML>=5.3.1',
        'tensorflow',
        'tensorboard',
    ],
    packages=['ddsp'],
    long_description=long_description,
    long_description_content_type='text.markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT'
)
