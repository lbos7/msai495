import cv2
import numpy as np

def CCL(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols, channels = img_gray.shape

    identified = False
    regions = []
    region_equivs = []

    for i in range(rows):
        for j in range(cols):
            left_neighbor_label = 0
            top_neighbor_label = 0
            equiv_set = -1
            if img_gray[i, j] != 0:
                if regions == []:
                    regions.append({(i, j)})
                elif i != 0:
                    for reg_num in range(len(regions)):
                        if (i - 1, j) in regions[reg_num]:
                            top_neighbor_label = reg_num + 1
                elif j != 0:
                    for reg_num in range(len(regions)):
                        if (i, j - 1) in regions[reg_num]:
                            left_neighbor_label = reg_num + 1
                
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
                            

                
            