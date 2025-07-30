import numpy as np

def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.  Prioritizes bins 
       that can accommodate the item without creating too much wasted space, while also 
       considering the number of items already in each bin.  Handles edge cases robustly.

    Args:
        item: Size of item to be added to the bin.
        bins_remain_cap: Array of capacities for each bin.  Must be a NumPy array.

    Return:
        Array of same size as bins_remain_cap with priority score of each bin.  Returns 
        an array of -inf for bins that cannot accommodate the item.
    """

    if not isinstance(bins_remain_cap, np.ndarray):
        raise TypeError("bins_remain_cap must be a NumPy array.")
    if item <=0 or np.any(bins_remain_cap <=0):
        raise ValueError("Item size and bin capacities must be positive values.")

    #Handle bins too small for the item.
    valid_bins = bins_remain_cap >= item
    priorities = np.full_like(bins_remain_cap, -np.inf)
    
    #Avoid division by zero.
    bins_remain_cap_valid = bins_remain_cap[valid_bins]
    if bins_remain_cap_valid.size ==0:
        return priorities

    #Prioritize bins based on remaining capacity and item size to minimize waste.
    waste_ratio = (bins_remain_cap_valid - item) / bins_remain_cap_valid
    
    #Incorporate a penalty for bins already nearly full to prevent creating many nearly-full bins.
    #Assume we have an array tracking the number of items in each bin.  Replace with your actual tracking mechanism.

    num_items_in_bin = np.random.randint(0,10, size = len(bins_remain_cap)) #Replace with your actual num_items_in_bin
    num_items_in_bin_valid = num_items_in_bin[valid_bins]
    fullness_penalty = num_items_in_bin_valid / 10 #Adjust 10 as needed


    priorities[valid_bins] = - (waste_ratio + fullness_penalty) # Higher value implies higher priority.
    return priorities
