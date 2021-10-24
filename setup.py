from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='supervised-product-matching',
   version='0.1',
   description='Neural network for product matching, aka classifying whether two product titles represent the same entity.',
   license="MIT",
   long_description=long_description,
   author='Jason Acheampong',
   author_email='jason.acheampong24@gmail.com',
   url="https://github.com/Mascerade/supervised-product-matching",
   packages=['supervised_product_matching', 'supervised_product_matching.model_architectures'],
   install_requires=['torch',
                     'transformers',
                     'nltk',
                     'numpy',
                     'scale_transformer_encoder @ git+https://github.com/Mascerade/scale-transformer-encoder@f684132c63cf7f8d771decd6fb560c9158ced361#egg=scale_transformer_encoder',
                     'character_bert @ git+https://github.com/Mascerade/character-bert@c44d0f1e7d2e822296a0578eecba52ddadd22d0e#egg=character_bert']
)