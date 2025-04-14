import cv2
import numpy as np

def CCL(img):
    label_img = np.copy(img)
    img_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    rows, cols = img_gray.shape

    regions = []
    region_equivs = []

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
                    regions[min(top_neighbor_label, left_neighbor_label)].add((i, j))

                    # Update region_equivs list to include region equivalances for second pass
                    if region_equivs == []:
                        region_equivs.append({top_neighbor_label, left_neighbor_label})
                    else:
                        for equiv_num in range(len(region_equivs)):
                            if left_neighbor_label in region_equivs[equiv_num] or top_neighbor_label in region_equivs[equiv_num]:
                                equiv_set = equiv_num
                        if equiv_set != -1:
                            region_equivs[equiv_set].update([top_neighbor_label, left_neighbor_label])
                        else:
                            region_equivs.append({left_neighbor_label, top_neighbor_label})

    # Removing redundant region labels, if necessary
    redundant_inds = []
    if region_equivs != []:
        for equiv in region_equivs:
            min_label = min(equiv)
            equiv.remove(min_label)
            for label in equiv:
                regions[min_label - 1] | regions[label - 1]
                redundant_inds.append(label - 1)
        redundant_inds.sort(reverse=True)
        for ind in redundant_inds:
            regions.pop(ind)

    # Generating different colors for labeling and updating pixels
    shades = np.linspace(0, 255, len(regions))
    for region_ind in range(len(regions)):
        for pixel in regions[region_ind]:
            img_gray[pixel[0], pixel[1]] = shades[region_ind]

    label_img[:, :, 0] = 0
    label_img[:, :, 1] = 0
    label_img[:, :, 2] = img_gray

    return label_img,len(regions)