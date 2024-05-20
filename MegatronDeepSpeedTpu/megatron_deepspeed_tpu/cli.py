"""Entrypoint for deepspeed_tpu that imports deepspeed_tpu module before running the deepspeed runner"""

from deepspeed.launcher.runner import main


def deepspeed_tpu_main():
    import megatron_deepspeed_tpu  # noqa
    main()
