import cv2
import numpy as np

def CCL(img, use_size_filter=False, filter_thresh=0):

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    # Blank image definition
    label_img = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Empty List Setup
    regions = []
    region_equivs = []
    redundant_inds = []
    remove_inds = []
    filtered_inds = []

    for i in range(rows):
        for j in range(cols):

            # Resetting label and equivalence variables each inner loop iteration
            left_neighbor_label = 0
            top_neighbor_label = 0
            equiv_set = -1

            # Check if the pixel is white
            if img_gray[i, j] != 0:

                # Update neighbor labels
                if i != 0:
                    for reg_num in range(len(regions)):
                        if (i - 1, j) in regions[reg_num]:
                            top_neighbor_label = reg_num + 1

                if j != 0:
                    for reg_num in range(len(regions)):
                        if (i, j - 1) in regions[reg_num]:
                            left_neighbor_label = reg_num + 1
                
                # Add pixel coords as a tuple to the corresponding set
                if top_neighbor_label == 0 and left_neighbor_label == 0:
                    regions.append({(i, j)})
                elif top_neighbor_label > 0 and left_neighbor_label == 0:
                    regions[top_neighbor_label - 1].add((i, j))
                elif top_neighbor_label == 0 and left_neighbor_label > 0:
                    regions[left_neighbor_label - 1].add((i, j))
                elif top_neighbor_label == left_neighbor_label:
                    regions[left_neighbor_label - 1].add((i, j))
                else:
                    regions[min(top_neighbor_label, left_neighbor_label) - 1].add((i, j))

                    if region_equivs != []:
                        for equiv_num in range(len(region_equivs)):
                            if region_equivs[equiv_num] & {top_neighbor_label, left_neighbor_label} != set():
                                region_equivs[equiv_num].update([top_neighbor_label, left_neighbor_label])
                                equiv_set = equiv_num
                        if equiv_set == -1:
                            region_equivs.append({top_neighbor_label, left_neighbor_label})
                    else:
                        region_equivs.append({top_neighbor_label, left_neighbor_label})

    # Combining region equivalences
    for i in range(len(region_equivs)):
        if i not in remove_inds:
            j = i + 1
            while j < len(region_equivs):
                if region_equivs[i] & region_equivs[j] != set():
                    region_equivs[i] = region_equivs[i] | region_equivs[j]
                    remove_inds.append(j)
                j += 1
    
    # Removing unnecessary region equivalences after combining together
    remove_inds.sort(reverse=True)
    for ind in remove_inds:
        region_equivs.pop(ind)

    # Removing redundant region labels, if necessary
    if region_equivs != []:
        for equiv in region_equivs:
            min_label = min(equiv)
            equiv.remove(min_label)
            for label in equiv:
                regions[min_label - 1] = regions[min_label - 1] | regions[label - 1]
                redundant_inds.append(label - 1)
        redundant_inds.sort(reverse=True)
        for ind in redundant_inds:
            regions.pop(ind)

    # Blank image setup
    label_img[:, :, 0] = 0
    label_img[:, :, 1] = 0
    label_img[:, :, 2] = 0

    # Filtering
    if use_size_filter:
        for i in range(len(regions)):
            if len(regions[i]) >= filter_thresh:
                filtered_inds.append(i)
        regions = [regions[passed_ind] for passed_ind in filtered_inds]

    # Generating different colors for labeling and updating pixels
    if len(regions) == 1:
        shades = [255]
    else:
        shades = np.linspace(70, 255, len(regions), dtype=np.uint8)
    for region_ind in range(len(regions)):
        for pixel in regions[region_ind]:
            label_img[pixel[0], pixel[1], 2] = shades[region_ind]

    return label_img,len(regions)