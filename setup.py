from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fin_gen_dl',
      version='0.1.0',
      install_requires=['neat-python==0.92', 'graphviz==0.16', 'matplotlib==3.4.1',
                        'svglib==1.1.0', 'reportlab==8.2.0', 'pika==1.2.0'],
      author="Matthew Siper",
      author_email="matt.quantheads.io@gmail.com",
      description="A package for evolving RNN topologies",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/matt-quant-heads-io/fin_gen_dl",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)