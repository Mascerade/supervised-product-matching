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
   packages=['supervised_product_matching'],
   install_requires=['torch',
                     'transformers',
                     'nltk',
                     'numpy',
                     'scale_transformer_encoder @ git+ssh://git@github.com/Mascerade/scale-transformer-encoder@v0.1#egg=scale_transformer_encoder',
                     'character_bert @ git+ssh://git@github.com/Mascerade/character-bert@v1.0#egg=character_bert']
)