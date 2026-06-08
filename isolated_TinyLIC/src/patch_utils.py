import torch

def patches_from_tensor(tensor, patch_size):
    """
    :return: list of all extracted patch tensors padded to size (a by b)
    """
    patch_height = patch_width = patch_size
    # Tensors appear to have size (output_dim, input_dim)
    patch_set = tensor.split(patch_height, -2)
    patch_set = [tens.split(patch_width, -1) for tens in patch_set]
    patch_set = flatten_list(patch_set)
    
    batched_patches = torch.cat(patch_set, dim=0)

    return batched_patches


def flatten_list(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def reconstruct_list(patches, recons1, recons2, padding_idxs, original_sizes):

    new_patches = []
    for i, tensor in enumerate(patches):
        # tensor = tensor.squeeze()
        if i in padding_idxs:
            og_size = original_sizes[padding_idxs.index(i)]
            tensor = tensor[:og_size[0], :og_size[1]]
        new_patches.append(tensor)

    new_new_patches = []
    i = 0
    for recons_len in recons1:
        stacked_tens = torch.cat([new_patches[i + j] for j in range(recons_len)], dim=0)
        new_new_patches.append(stacked_tens)
        i += recons_len

    # new_new_new_patches = []
    # i = 0
    # for recons_len in recons1:
    #     stacked_tens = torch.cat([new_new_patches[i + j] for j in range(recons_len)], dim=1)
    #     new_new_new_patches.append(stacked_tens)
    #     i += recons_len

    return new_new_new_patches