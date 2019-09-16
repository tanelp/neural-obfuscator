import setuptools

setuptools.setup(
    name="neural-obfuscator",
    version="0.1",
    author="Tanel Pärnamaa",
    author_email="tanel.parnamaa@gmail.com",
    description="A library for anonymizing the identities in an image by swapping faces to the ones that have never existed before.",
    url="https://github.com/tanelp/neural-obfuscator",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
