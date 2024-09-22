import os, argparse
import numpy as np
import tqdm
import openslide

import utils.dataset as dataset

def retrieve_file(dir, name_without_extension):
    for file_ in os.listdir(dir):
        if os.path.split(file_)[0] == name_without_extension:
            return os.path.join(dir, file_)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True, type=str, help="path to directory with regions annotations")
    parser.add_argument("--wsi_path", required=True, type=str, help="path to directory with WSI files")
    parser.add_argument("--output_path", required=True, type=str, help="path to output directory")
    parser.add_argument("--mask_level", required=True, type=int, help="level to create tissue mask")
    parser.add_argument("--patch_level", required=True, type=int, help="level to extract patches")
    parser.add_argument("--patch_size", required=True, type=int, help="size of patches (width and heigth)")
    parser.add_argument("--max_length", default=256, type=int, help="maximum length of a sequence")
    parser.add_argument("--min_length", default=200, type=int, help="minimum length of a sequence")
    parser.add_argument("--min_tissue_ratio", required=True, type=float, help="threshold of tissue in patch")
    args = parser.parse_args()

    for stain in os.listdir(args.annotations_path):
        print(stain)

        for annot_file in tqdm.tqdm(os.listdir(os.path.join(args.annotations_path, stain)), ncols=50):

            slide_name = os.path.split(annot_file)[0]

            slide = retrieve_file(args.wsi_path, slide_name)
            if slide is None:
                continue

            # open slide file
            slide = openslide.OpenSlide(slide)

            # dir to save patches
            slide_dir = os.path.join(args.output_path, slide_name)
            if not os.path.isdir(slide_dir):
                os.makedirs(slide_dir, exist_ok=True)

            # tissue mask of the whole slide
            tissue_mask = dataset.get_tissue_mask(slide, args.mask_level)
            
            # load regions
            regions = dataset.load_contours_from_json(os.path.join(args.annotations_path, stain, annot_file))
            for id_, ctr in regions.items():
                
                region_dir = os.path.join(slide_dir, id_)
                os.makedirs(region_dir, exist_ok=True)
                
                stain_dir = os.path.join(region_dir, stain)
                os.makedirs(stain_dir, exist_ok=True)

                # coordinates of region's bounding-box at level 0
                x0 = ctr[..., 0].min()
                y0 = ctr[..., 1].min()
                x1 = ctr[..., 0].max()
                y1 = ctr[..., 1].max()

                # creates region mask
                region_mask = dataset.create_object_mask(slide, ctr, args.mask_level)

                # combine region and tissue masks
                region_mask = np.logical_and(region_mask, tissue_mask)

                W, H = slide.level_dimensions[args.patch_level]

                # iterate over WSI to extract patches within the region
                region_patches = []
                for x in range(0, W - args.patch_size, args.patch_size):
                    for y in range(0, H - args.patch_size, args.patch_size):

                        # coordinates of patch at level 0
                        x, y = dataset.scale_coordinates(slide, (x, y), args.patch_level, 0)

                        # check patch upper-left corner is within region
                        if not (x > x0 and x < x1 and y > y0 and y < y1):
                            continue

                        # coordinates of patch at mask level
                        x, y = dataset.scale_coordinates(slide, (x, y), 0, args.patch_level)
                        xbr, ybr = dataset.scale_coordinates(slide,
                                                             (x + args.patch_size, y + args.patch_size),
                                                             args.patch_level,
                                                             args.mask_level)
                        x, y = dataset.scale_coordinates(slide, (x, y), args.patch_level, args.mask_level)

                        # tissue ratio
                        patch_mask = region_mask[y:ybr, x:xbr]
                        tissue_ratio = patch_mask.sum() / patch_mask.size
                        if tissue_ratio >= args.min_tissue_ratio:
                            
                            # coordinates of patch at level 0
                            x, y = dataset.scale_coordinates(slide, (x, y), args.mask_level, 0)
                            
                            # store patch location
                            region_patches.append((x,y))

                # split region list to max_length * (equal length lists) and shuffle
                region_patches = np.array_split(region_patches, args.max_length)
                region_patches = [np.random.permutation(i).tolist() for i in region_patches]
                
                # creates list of samples
                samples = []
                while all(region_patches):
                    one_sample = [i.pop() for i in region_patches]
                    samples.append(one_sample)

                # save sequences on disk
                for i, patches in enumerate(samples):

                    # check there are enough patches
                    if len(patches) < args.min_length:
                        continue

                    # extract and save the patches in a numpy file
                    fn = lambda p : slide.read_region(p, args.patch_level, (args.patch_size, args.patch_size)).convert('RGB')
                    patches = np.stack(list(map(fn, patches)), axis=0)
                    path_ = os.path.join(stain_dir, '{}.npy'.format(i))
                    np.save(path_, patches)
    
    print("done")
