import setuptools

setuptools.setup(
    name="megatron_deepspeed_tpu",
    version="0.0.3",
    description="An adaptor for deepspeed and megatron-deepspeed on Sophgo tpu",
    packages=['megatron_deepspeed_tpu', "tpu_workarounds"],
    install_package_data=True,
    include_package_data=True,
    install_requires=[
        # "deepspeed==0.13.5",
        # "torch==2.1.0",
        "nnmoduletools>=0.1.1",
        "netifaces", # for dist network interface
    ],
    entry_points={
        "console_scripts": [
            "deepspeed_tpu=megatron_deepspeed_tpu.cli:deepspeed_tpu_main",
            "ds_tpu=megatron_deepspeed_tpu.cli:deepspeed_tpu_main",
        ],
    },
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
