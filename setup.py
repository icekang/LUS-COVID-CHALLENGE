import setuptools

setuptools.setup(
    name="icekang",  # Replace with your own username
    version="0.1",
    author="MLO - EPFL",
    author_email="naravich.chutisilp@epfl.ch",
    description="DeepChest lung ultrasound challenge",
    url="https://github.com/epfl-iglobalhealth/LUS-COVID-CHALLENGE",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "einops>=0.3.0",
        "ml_collections",
        "pandas>=1.0.1",
        "plotly>=4.14.3",
        "rich>=9.8.2",
        "scikit-image>=0.16.2",
        "scikit-learn>=0.22.1",
        "termcolor>=1.1.0",
        "wandb>=0.10.14",
        "torch",
        "torchvision",
        "rich",
        "tqdm"
    ],
)
