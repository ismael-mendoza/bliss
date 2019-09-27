import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import simulated_datasets_lib

# Tile images

# The next two functions copied from
# https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
def _extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches

def _reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img


def tile_images(images, subimage_slen, step, return_tile_coords = False):
    # breaks up a large image into smaller patches,
    # of size subimage_slen x subimage_slen
    # NOTE: input and output are torch tensors, not numpy arrays
    #       (need the unfold command from torch)

    # image should be batchsize x 1 x slen x slen
    assert len(images.shape) == 4

    image_xlen = images.shape[2]
    image_ylen = images.shape[3]

    # my tile coords doens't work otherwise ...
    assert (image_xlen - subimage_slen) % step == 0
    assert (image_ylen - subimage_slen) % step == 0

    images_batched = _extract_patches_2d(images,
                                        patch_shape = [subimage_slen, subimage_slen],
                                        step = [step, step],
                                        batch_first = True).reshape(-1, 1, subimage_slen, subimage_slen)


    if return_tile_coords:
        nx_patches = ((image_xlen - subimage_slen) // step) + 1
        ny_patches = ((image_ylen - subimage_slen) // step) + 1
        n_patches = nx_patches * ny_patches

        return_coords = lambda i : [(i // ny_patches) * step,
                                    (i % ny_patches) * step]

        tile_coords = torch.LongTensor([return_coords(i) \
                                        for i in range(n_patches)]).to(device)

        return images_batched, tile_coords, nx_patches, ny_patches, n_patches
    else:
        return images_batched

def get_params_in_patches(tile_coords, locs, fluxes, slen, subimage_slen,
                            edge_padding = 0,
                            sort_locs = False):
    n_patches = tile_coords.shape[0] # number of patches in a full image
    fullimage_batchsize = locs.shape[0]
    assert fullimage_batchsize == fluxes.shape[0]

    subimage_batchsize = n_patches * fullimage_batchsize

    max_stars = locs.shape[1]
    assert max_stars == fluxes.shape[1]

    tile_coords = tile_coords.unsqueeze(0).unsqueeze(2).float()
    locs = locs * (slen - 1)
    which_locs_array = (locs.unsqueeze(1) > tile_coords + edge_padding) & \
                        (locs.unsqueeze(1) < tile_coords + subimage_slen - 1 - edge_padding)
    which_locs_array = (which_locs_array[:, :, :, 0] * which_locs_array[:, :, :, 1]).float()

    subimage_locs = \
        (which_locs_array.unsqueeze(3) * locs.unsqueeze(1) - \
            tile_coords - edge_padding).view(subimage_batchsize, max_stars, 2) / \
                (subimage_slen - 1 - 2 * edge_padding)
    subimage_locs = torch.relu(subimage_locs) # by subtracting off, some are negative now; just set these to 0

    subimage_fluxes = \
        (which_locs_array * fluxes.unsqueeze(1)).view(subimage_batchsize, max_stars)

    # sort locs so all the zeros are at the end
    if sort_locs:
        sort_perm = torch.argsort(torch.sum(subimage_locs**2, dim = 2),
                                            descending=True)
        seq_tensor = torch.LongTensor([[i] for i in range(subimage_batchsize)])
        subimage_locs = subimage_locs[seq_tensor, sort_perm]
        subimage_fluxes = subimage_fluxes[seq_tensor, sort_perm]
        is_on_array = subimage_fluxes > 0
    else:
        is_on_array = which_locs_array.view(subimage_batchsize, max_stars).type(torch.bool).to(device)

    n_stars = is_on_array.float().sum(dim = 1).type(torch.LongTensor).to(device)

    return subimage_locs, subimage_fluxes, n_stars, is_on_array
