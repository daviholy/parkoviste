from torch import zeros, nn, stack


class Common:
    args = None

    @classmethod
    def commonArguments(cls, parser):
        parser.add_argument('--DEBUG', type=bool, default=False,
                            help='decides if script will run in debug mode (prints to stdout)')
        cls.args = parser.parse_args()
        return cls.args

    @staticmethod
    def _collate_fn_pad(batch):
        """
        Takes images as tensors and labels as input, finds highest and widest size of an image than pad smaller images.

        :param batch: list of images and labels [img_as_tensor, 'label', ....]
        :return: returns tuple where ([padded_img_as_tensors,..], ('label',..))
        :rtype: (list, tuple)
        """
        imgs, labels = zip(*batch)
        # get max width and height
        h, w = zip(*[list(t[0].size()) for t in imgs])
        max_h, max_w = max(h), max(w)

        padded_imgs = zeros(len(batch), 1, max_h, max_w)
        # padding
        for x in range(len(batch)):
            img = batch[x][0]
            pad_h = max_h - img[0].size(0)
            pad_w = max_w - img[0].size(1)

            pad_l = int(pad_w / 2)  # left
            pad_r = pad_w - pad_l  # right
            pad_t = int(pad_h / 2)  # top
            pad_b = pad_h - pad_t  # bottom
            pad = nn.ZeroPad2d((pad_l, pad_r, pad_t, pad_b))
            padded_imgs[x] = pad(img)

        return padded_imgs, stack(labels)

    @classmethod
    def debug(cls, name):
        def decorator(func):
            def inner(*arg):
                if cls.args.DEBUG:
                    print(name)
                    func(*arg)
                    print()
                return

            return inner

        return decorator
