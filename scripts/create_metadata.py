from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/work/dlclarge2/ferreira-dinov2/imagenet", extra="/work/dlclarge2/ferreira-dinov2/imagenet-extra")
    dataset.dump_extra()