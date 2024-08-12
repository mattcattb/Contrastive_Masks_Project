from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset,IMG_EXTENSIONS
from torchvision import datasets
import torch

import tensorflow.compat.v1 as tf

from utils.img import ImageCoder

class SSLTFDataset(VisionDataset):
    def __init__(self, root: str, extensions = IMG_EXTENSIONS, transform = None):
        self.root = root
        self.transform = transform
        self.samples = make_dataset(self.root, extensions = extensions) #Pytorch 1.9+
        self.coder = ImageCoder()

    def _is_cmyk(self,filename):
        """Determine if file contains a CMYK JPEG format image.

        Args:
        filename: string, path of the image file.

        Returns:
        boolean indicating if the image is a JPEG encoded with CMYK color space.
        """
        # File list from:
        # https://github.com/cytsai/ilsvrc-cmyk-image-list
        cmyk_excluded = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                       'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                       'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                       'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                       'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                       'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                       'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                       'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                       'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                       'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                       'n07583066_647.JPEG', 'n13037406_4650.JPEG']
        return filename.split('/')[-1] in cmyk_excluded

    def _is_png(self,filename):
        """Determine if a file contains a PNG format image.

        Args:
        filename: string, path of the image file.

        Returns:
        boolean indicating if the image is a PNG.
        """
        # File list from:
        # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
        return 'n02105855_2933.JPEG' in filename

    def _process_image(self, filename):

        """Process a single image file.
        Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        fh_scales: Felzenzwalb-Huttenlocher segmentation scales.
        fh_min_sizes: Felzenzwalb-Huttenlocher min segment sizes.
        Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
        """
    
        # Read the image file.
        image_data = tf.gfile.GFile(filename, 'rb').read()

        #ZZimport ipdb;ipdb.set_trace()
        # Clean the dirty data.
        if self._is_png(filename):
            # 1 image is a PNG.
            print('Converting PNG to JPEG for %s' % filename)
            image_data = self.coder.png_to_jpeg(image_data)
        elif self._is_cmyk(filename):
            # 22 JPEG images are in CMYK colorspace.
            print('Converting CMYK to RGB for %s' % filename)
            image_data = self.coder.cmyk_to_rgb(image_data)

        # Decode the RGB JPEG.
        image = self.coder.decode_jpeg(image_data)

        # Check that image converted to RGB
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        return image

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        
        # Load Image
        image = self._process_image(path)
        label = 0
        
        return (torch.tensor(image).permute(2,0,1),label,path)

    def __len__(self) -> int:
        return len(self.samples)