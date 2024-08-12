import os
import random

from data import SSLTFDataset

from masks import PatchMaskGenerator, SAMMaskGenerator, FHMaskGenerator

from tensorboardX import SummaryWriter
import torchshow as ts
def view_masks():
    pg = PatchMaskGenerator()
    sg = SAMMaskGenerator()
    fg = FHMaskGenerator()

    logdir = "logs/"
    writer = SummaryWriter(logdir)

    images_dir = "/media/mattyb/UBUNTU 22_0/datasets/imagenet_strawberries/images/train"
    dataset = SSLTFDataset(images_dir)

    for i in range(3):
        idx = random.randint(0, len(dataset) - 1)
        img, label, path = dataset[idx]
        # img is h, w, c

        pg_mask = pg.create_mask(img)
        sg_mask = sg.create_mask(path)
        fg_mask = fg.create_mask(img)
        ts.show(pg_mask)
        ts.show(sg_mask)
        ts.show(fg_mask)
        img = img.numpy().transpose((1,2,0)) # (h, w, c)

        writer.add_image(f'Image_{i}/Original', img, i, dataformats='HWC')
        writer.add_image(f'Image_{i}/Patch_Mask', pg_mask, i, dataformats='HWC')
        writer.add_image(f'Image_{i}/SAM_Mask', sg_mask, i, dataformats='HWC')
        writer.add_image(f'Image_{i}/FH_Mask', fg_mask, i, dataformats='HWC')

    writer.close()

if __name__ == "__main__":

    view_masks()