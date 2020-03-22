import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


sys.path.insert(0, '../')
from GalaxyModel.src import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _check_psf(psf, slen):
    # first dimension of psf is number of bands
    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    # dimension of the psf should be odd
    psf_slen = psf.shape[2]
    assert psf.shape[1] == psf_slen
    assert (psf_slen % 2) == 1
    # same for slen
    assert (slen % 2) == 1


def _trim_psf(psf, slen):
    """
    Crop the psf to length slen x slen,
    centered at the middle
    :param psf:
    :param slen:
    :return:
    """
    #

    _check_psf(psf, slen)
    psf_slen = psf.shape[2]
    psf_center = (psf_slen - 1) / 2

    assert psf_slen >= slen

    r = np.floor(slen / 2)
    l_indx = int(psf_center - r)
    u_indx = int(psf_center + r + 1)

    return psf[:, l_indx:u_indx, l_indx:u_indx]


def _expand_psf(psf, slen):
    """
    Pad the psf with zeros so that it is size slen,
    :param psf:
    :param slen:
    :return:
    """

    _check_psf(psf, slen)
    psf_slen = psf.shape[2]

    assert psf_slen <= slen

    psf_expanded = torch.zeros((n_bands, slen, slen))

    offset = int((slen - psf_slen) / 2)

    psf_expanded[:, offset:(offset + psf_slen), offset:(offset + psf_slen)] = psf

    return psf_expanded


def _get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset:(offset + 1), -offset:(offset + 1)]
    return (torch.Tensor(np.dstack((y, x))) / offset).to(device)


def plot_one_star(slen, locs, psf, cached_grid=None):
    """

    :param slen:
    :param locs: is batchsize x 2: takes values between 0 and 1
    :param psf: is a slen x slen tensor
    :param cached_grid:
    :return:
    """

    assert len(psf.shape) == 3
    n_bands = psf.shape[0]

    batchsize = locs.shape[0]
    assert locs.shape[1] == 2

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    # scale locs so they take values between -1 and 1 for grid sample
    locs = (locs - 0.5) * 2
    grid_loc = grid.view(1, slen, slen, 2) - locs[:, [1, 0]].view(batchsize, 1, 1, 2)

    star = F.grid_sample(psf.expand(batchsize, n_bands, -1, -1), grid_loc, align_corners=True)

    # normalize so one star still sums to 1
    return star  # / star.sum(3, keepdim=True).sum(2, keepdim=True)


def plot_multiple_stars(slen, locs, n_stars, fluxes, psf, cached_grid=None):
    """

    :param slen:
    :param locs: is (batchsize x max_stars x (x_loc, y_loc))
    :param n_stars: batchsize
    :param fluxes: is batchsize x n_bands x max_stars
    :param psf: is a n_bands x slen x slen tensor
    :param cached_grid: Grid where the stars should be plotted with shape (slen x slen)
    :return:
    """

    n_bands = psf.shape[0]

    batchsize = locs.shape[0]
    assert locs.shape[2] == 2

    assert fluxes.shape[0] == locs.shape[0]
    assert fluxes.shape[1] == locs.shape[1]
    assert fluxes.shape[2] == n_bands
    assert len(n_stars) == batchsize
    assert len(n_stars.shape) == 1

    assert max(n_stars) <= locs.shape[1]

    if cached_grid is None:
        grid = _get_mgrid(slen)
    else:
        assert cached_grid.shape[0] == slen
        assert cached_grid.shape[1] == slen
        grid = cached_grid

    stars = 0.  # torch.zeros((batchsize, 1, slen, slen)).to(device)

    for n in range(max(n_stars)):
        is_on_n = (n < n_stars).float()
        locs_n = locs[:, n, :] * is_on_n.unsqueeze(1)
        fluxes_n = fluxes[:, n, :]

        one_star = plot_one_star(slen, locs_n, psf, cached_grid=grid)

        stars += one_star * (is_on_n.unsqueeze(1) * fluxes_n).view(batchsize, n_bands, 1, 1)

    return stars


def _draw_pareto(f_min, alpha, shape):
    uniform_samples = torch.rand(shape).to(device)

    return f_min / (1 - uniform_samples) ** (1 / alpha)


def _draw_pareto_maxed(f_min, f_max, alpha, shape):
    # draw pareto conditioned on being less than f_max

    pareto_samples = _draw_pareto(f_min, alpha, shape)

    while torch.any(pareto_samples > f_max):
        indx = pareto_samples > f_max
        pareto_samples[indx] = \
            _draw_pareto(f_min, alpha, torch.sum(indx))

    return pareto_samples


class GalaxySimulator:
    def __init__(self, slen, background, decoder_file):
        """

        :param slen:
        :param background:
        :param decoder_file: Decoder file where decoder network trained on individual galaxy images is.
        """

    def draw_image_from_params(self, locs, params, n_galaxies):


class StarSimulator:
    def __init__(self, psf, slen, background, transpose_psf):
        assert len(psf.shape) == 3

        assert len(background.shape) == 3
        assert background.shape[0] == psf.shape[0]
        assert background.shape[1] == slen
        assert background.shape[2] == slen
        self.background = background

        self.n_bands = psf.shape[0]
        self.psf_og = psf

        # side length of the image
        self.slen = slen

        # get psf shape to match image shape
        # if slen is even, we still make psf dimension odd.
        #   otherwise, the psf won't have a peak in the center pixel.
        _slen = slen + ((slen % 2) == 0) * 1
        if slen >= self.psf_og.shape[-1]:
            self.psf = _expand_psf(self.psf_og, _slen).to(device)
        else:
            self.psf = _trim_psf(self.psf_og, _slen).to(device)

        if transpose_psf:
            self.psf = self.psf.transpose(1, 2)

        # TODO:
        # should we then upsample??

        # normalize
        # self.psf = self.psf / torch.sum(self.psf)

        self.cached_grid = _get_mgrid(slen).to(device)

    def draw_image_from_params(self, locs, fluxes, n_stars,
                               add_noise=True):
        """

        :param locs:
        :param fluxes:
        :param n_stars:
        :param add_noise:
        :return: `images`, torch.Tensor of shape (n_images x n_bands x slen x slen)

        NOTE: The different sources in `images` are already aligned between bands.
        """
        images_mean = \
            plot_multiple_stars(self.slen, locs, n_stars, fluxes,
                                self.psf, self.cached_grid) + \
            self.background[None, :, :, :]

        # add noise
        if add_noise:
            if torch.any(images_mean <= 0):
                print('warning: image mean less than 0')
                images_mean = images_mean.clamp(min=1.0)

            images = (torch.sqrt(images_mean) * torch.randn(images_mean.shape).to(device)
                      + images_mean)
        else:
            images = images_mean

        return images


class StarsDataset(Dataset):

    def __init__(self, psf, n_images,
                 slen,
                 max_stars,
                 mean_stars,
                 min_stars,
                 f_min,
                 f_max,
                 background,
                 alpha,
                 transpose_psf=False,
                 add_noise=True):
        """

        :param psf:
        :param n_images: same as batchsize.
        :param slen:
        :param max_stars: Default value 1500
        :param mean_stars: Default value 1200
        :param min_stars: Default value 0
        :param f_min:
        :param f_max:
        :param background:
        :param alpha:
        :param transpose_psf:
        :param add_noise:
        """

        self.slen = slen
        self.n_bands = psf.shape[0]

        self.simulator = StarSimulator(psf, slen, background, transpose_psf)
        self.background = background[None, :, :, :]

        # image parameters
        self.max_stars = max_stars
        self.mean_stars = mean_stars
        self.min_stars = min_stars
        self.add_noise = add_noise

        # TODO: make this an argument
        self.draw_poisson = True

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max

        self.alpha = alpha

        # dataset parameters
        self.n_images = n_images  # = batchsize.

        # set the first batch of data.
        self.set_params_and_images()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        return {'image': self.images[idx],
                'background': self.background[0],
                'locs': self.locs[idx],
                'fluxes': self.fluxes[idx],
                'n_stars': self.n_stars[idx]}

    def draw_batch_parameters(self, batchsize, return_images=True):
        # sample number of stars
        if self.draw_poisson:
            n_stars = np.random.poisson(self.mean_stars, batchsize)
        else:
            n_stars = np.random.choice(np.arange(self.min_stars, self.max_stars + 1),
                                       batchsize)

        n_stars = torch.Tensor(n_stars).clamp(max=self.max_stars,
                                              min=self.min_stars).type(torch.LongTensor).to(device)
        is_on_array = utils.get_is_on_from_n_stars(n_stars, self.max_stars)

        # sample locations
        locs = torch.rand((batchsize, self.max_stars, 2)).to(device) * is_on_array.unsqueeze(2).float()

        # sample fluxes
        base_fluxes = _draw_pareto_maxed(self.f_min, self.f_max, alpha=self.alpha,
                                         shape=(batchsize, self.max_stars))

        if self.n_bands > 1:
            colors = torch.randn(batchsize, self.max_stars, self.n_bands - 1).to(device) * 0.15 + 0.3

            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(2)

            fluxes = torch.cat((base_fluxes.unsqueeze(2), _fluxes), dim=2) * is_on_array.unsqueeze(2).float()
        else:
            fluxes = (base_fluxes * is_on_array.float()).unsqueeze(2)

        if return_images:
            images = self.simulator.draw_image_from_params(locs, fluxes, n_stars,
                                                           add_noise=self.add_noise)
            return locs, fluxes, n_stars, images

        else:
            return locs, fluxes, n_stars

    def set_params_and_images(self):
        """
        Images is now attached to a device.
        :return:
        """
        self.locs, self.fluxes, self.n_stars, self.images = \
            self.draw_batch_parameters(self.n_images, return_images=True)


def load_dataset_from_params(psf, data_params,
                             n_images,
                             background,
                             transpose_psf=False,
                             add_noise=True):
    # data parameters
    slen = data_params['slen']

    f_min = data_params['f_min']
    f_max = data_params['f_max']
    alpha = data_params['alpha']

    max_stars = data_params['max_stars']
    mean_stars = data_params['mean_stars']
    min_stars = data_params['min_stars']

    # draw data
    return StarsDataset(psf,
                        n_images,
                        slen=slen,
                        f_min=f_min,
                        f_max=f_max,
                        max_stars=max_stars,
                        mean_stars=mean_stars,
                        min_stars=min_stars,
                        alpha=alpha,
                        background=background,
                        transpose_psf=transpose_psf,
                        add_noise=add_noise)
