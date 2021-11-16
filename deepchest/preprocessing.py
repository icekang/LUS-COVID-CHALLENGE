import re

import torch

from deepchest import dataset, utils


def independent_dropout(dropout_probability):
    def pp(images, sites, mask):
        if dropout_probability > 0.0:
            mask = torch.empty_like(mask).bernoulli_(1 - dropout_probability) * mask
        return images, sites, mask

    return pp


def keep_one_image_per_site(keep_first=False):
    def pp(images, sites, mask):
        batch_size, max_images_per_patient = sites.shape
        num_sites = len(dataset.SITE_MAPPING)

        sites_indices = torch.arange(num_sites, device=sites.device)

        # batch, num_sites, max_images_per_patient
        mask_per_site = sites[:, None, :] == sites_indices[None, :, None]

        # batch, num_sites
        count_sites = mask_per_site.sum(dim=-1)

        # batch_size, num_sites
        if not keep_first:
            selected_image_idx_per_site = (
                torch.randint(9999999, (batch_size, num_sites), device=sites.device).long()
                % count_sites.clip(min=1)  # clip to one to avoid modulo by 0.
            ).long()
        else:
            selected_image_idx_per_site = torch.zeros(
                (batch_size, num_sites), device=sites.device
            ).long()

        # batch, num_sites, max_images_per_patient
        image_order_per_site = utils.exclusive_cumsum(mask_per_site.long(), dim=-1)
        selected_image = (
            image_order_per_site == selected_image_idx_per_site[:, :, None]
        ) & mask_per_site
        new_mask = (selected_image.sum(dim=1) > 0).float() * mask

        return images, sites, new_mask

    return pp


def keep_single_image(images, sites, mask):
    batch_size = images.shape[0]
    num_images_per_patients = mask.sum(axis=1).long()

    # Pick a random image per patient.
    # Generates a large integer and take modulo the number of images to have a pseudo
    # uniform pick.
    selected_indices = (
        torch.randint(100000 * mask.shape[1], (batch_size,), device=images.device).long()
        % num_images_per_patients
    ).long()

    # Extract the selected elements
    batch_range = torch.arange(batch_size, device=images.device, dtype=torch.long)
    images = images[batch_range, selected_indices]
    sites = sites[batch_range, selected_indices]
    mask = mask[batch_range, selected_indices]

    # Add the lonely dim again
    images = images[:, None]
    sites = sites[:, None]
    mask = mask[:, None]

    return images, sites, mask


def random_sites(images, sites, mask):
    sites = torch.randint(
        1,
        len(dataset.SITE_MAPPING),
        size=sites.shape,
        dtype=sites.dtype,
        device=sites.device,
    )
    return images, sites, mask


def make_keep_sites_from_regex(regex):
    return set(
        site_name
        for site_name, _ in dataset.SITE_MAPPING.items()
        if re.fullmatch(regex, site_name) is not None
    )


def filter_sites(regex=False):
    keep_sites = make_keep_sites_from_regex(regex)
    print("Filter sites keeping:", ", ".join(sorted(list(keep_sites))))
    keep_sites_indicators = torch.Tensor(
        [site_name in keep_sites for site_name in dataset.SITE_MAPPING]
    )
    if keep_sites_indicators.sum().item() == 0:
        raise ValueError(f"Regex '{regex}' to filter sites did not match any.")

    def pp(images, sites, mask):
        device = sites.device
        num_sites = len(dataset.SITE_MAPPING)
        sites_indices = torch.arange(num_sites, device=device)

        # batch, num_sites, max_images_per_patient
        mask_per_site = sites[:, None, :] == sites_indices[None, :, None]
        mask_per_site = mask_per_site * keep_sites_indicators.to(device)[None, :, None]
        new_mask = mask_per_site.max(dim=1).values

        return images, sites, new_mask

    return pp


def merge_sites(*regexes):
    """Replaces sites in a group (matching regex) by first site. Used to have coarser grouping of sites."""
    sites_groupings = [make_keep_sites_from_regex(regex) for regex in regexes]
    flatten_sites_groupings = [g for gs in sites_groupings for g in gs]
    if len(flatten_sites_groupings) != len(set(flatten_sites_groupings)):
        raise ValueError(f"Some sites are assigned to more than one grouping: {sites_groupings}")
    if any(len(g) == 0 for g in sites_groupings):
        raise ValueError(f"Some groupings were empty: {sites_groupings} for {list(regexes)}.")
    print(f"Grouped the following sites: {sites_groupings}.")

    # Create a lookup tensor that maps each site in the group to the smallest index of the group.
    # By default the lookup is identity.
    lookup_tensor = torch.arange(len(dataset.SITE_MAPPING), dtype=torch.long)
    for group in sites_groupings:
        group_idx = min(dataset.SITE_MAPPING[site] for site in group)
        for site in group:
            lookup_tensor[dataset.SITE_MAPPING[site]] = group_idx

    def pp(images, sites, mask):
        device = sites.device
        lookup_tensor_ = lookup_tensor.to(device)
        print(sites.dtype)
        new_sites = lookup_tensor_[sites]
        return images, new_sites, mask

    return pp


_available_pp = {
    "identity": lambda a, b, c: (a, b, c),
    "independent_dropout": independent_dropout,
    "keep_one_image_per_site": keep_one_image_per_site,
    "keep_single_image": keep_single_image,
    "random_sites": random_sites,
    "filter_sites": filter_sites,
    "merge_sites": merge_sites,
}


def make_preprocessing_fn(pp_str):
    pp_str = pp_str or "identity"
    fns = [eval(exp, {}, _available_pp) for exp in pp_str.split(">>")]

    def pp(images, sites, mask):
        for f in fns:
            images, sites, mask = f(images, sites, mask)
        return images, sites, mask

    return pp
