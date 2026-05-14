import numpy as np

def create_balanced_indices(stock_list, stock_to_id, stock_data, stock_window_counts):
    """
    Balanced sampling: each stock has equal representation
    """
    counts = list(stock_window_counts.values())
    median_count = int(np.median(counts))

    # Pre-allocate numpy arrays for speed and memory efficiency
    total_samples = len(stock_data) * median_count
    stock_ids = np.zeros(total_samples, dtype=np.uint8)
    window_indices = np.zeros(total_samples, dtype=np.uint32)

    for i, stock in enumerate(stock_list):
        stock_id = stock_to_id[stock]
        stock_indices = stock_data[stock]["indices"]
        count = len(stock_indices)

        if count >= median_count:
            sampled = np.random.choice(
                stock_indices, size=median_count, replace=False
            )
        else:
            sampled = np.random.choice(
                stock_indices, size=median_count, replace=True
            )

        start = i * median_count
        end = (i + 1) * median_count
        stock_ids[start:end] = stock_id
        window_indices[start:end] = sampled

    return stock_ids, window_indices

def create_unbalanced_indices(stock_list, stock_to_id, stock_data):
    """
    Unbalanced sampling: concatenate all windows
    """
    all_stock_ids = []
    all_window_indices = []

    for stock in stock_list:
        stock_id = stock_to_id[stock]
        indices = stock_data[stock]["indices"]

        all_stock_ids.append(np.full(len(indices), stock_id, dtype=np.uint8))
        all_window_indices.append(np.array(indices, dtype=np.uint32))

    stock_ids = np.concatenate(all_stock_ids)
    window_indices = np.concatenate(all_window_indices)

    return stock_ids, window_indices

def create_sample_indices(balance_stocks, stock_list, stock_to_id, stock_data, stock_window_counts):
    """
    Create sample indices with optional balancing
    Returns numpy arrays for stock_ids and window_indices
    """
    if balance_stocks:
        stock_ids, window_indices = create_balanced_indices(
            stock_list, stock_to_id, stock_data, stock_window_counts
        )
    else:
        stock_ids, window_indices = create_unbalanced_indices(
            stock_list, stock_to_id, stock_data
        )

    # Shuffle using a shared permutation to avoid list-to-array spikes
    N = len(stock_ids)
    perm = np.random.permutation(N)
    return stock_ids[perm], window_indices[perm]
