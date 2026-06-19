
import numpy as np


def _l2_normalize(arr, axis=1, eps=1e-12):
    """
    Row-wise L2 normalization using NumPy only.
    """
    arr = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def _cosine_similarity(a, b=None, eps=1e-12):
    """
    Cosine similarity using NumPy only.

    Parameters
    ----------
    a : np.ndarray, shape (N, d) or (d,)
    b : np.ndarray, shape (M, d) or (d,), optional

    Returns
    -------
    sim : np.ndarray, shape (N, M)
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)

    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=float)
        if b.ndim == 1:
            b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    a_norm = np.maximum(a_norm, eps)
    b_norm = np.maximum(b_norm, eps)

    sim = (a @ b.T) / (a_norm @ b_norm.T)
    return sim


def mmr_sample(
    X,
    k,
    query=None,
    lambda_param=0.5,
    existing_indices=None,
    existing_vectors=None,
    exclude_existing_from_candidates=True,
    normalize_vectors=True,
    return_scores=False,
):
    """
    MMR-based sampling with support for pre-selected existing samples.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Candidate vectors.
    k : int
        Number of NEW samples to select.
    query : np.ndarray, shape (d,) or (1, d), optional
        Query vector. If None, centroid of X is used.
    lambda_param : float, default=0.5
        Trade-off between relevance and diversity.
        Higher => more relevance, lower => more diversity.
    existing_indices : list[int], optional
        Indices of already-selected samples inside X.
    existing_vectors : np.ndarray, shape (M, d), optional
        Already-selected sample vectors not necessarily inside X.
    exclude_existing_from_candidates : bool, default=True
        If True, existing_indices are excluded from new selection candidates.
    normalize_vectors : bool, default=True
        Whether to L2-normalize vectors before cosine similarity.
    return_scores : bool, default=False
        If True, also return selected scores.

    Returns
    -------
    selected_indices : list[int]
        Indices of newly selected samples from X.
    selected_scores : list[float], optional
        Returned only if return_scores=True.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (N, d)")
    N, d = X.shape

    existing_indices = [] if existing_indices is None else list(existing_indices)

    if existing_vectors is not None:
        existing_vectors = np.asarray(existing_vectors, dtype=float)
        if existing_vectors.ndim == 1:
            existing_vectors = existing_vectors.reshape(1, -1)
        if existing_vectors.shape[1] != d:
            raise ValueError("existing_vectors must have shape (M, d) with same d as X")

    if query is None:
        query_vec = X.mean(axis=0, keepdims=True)
    else:
        query_vec = np.asarray(query, dtype=float)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != d:
            raise ValueError("query must have dimension d matching X")

    # normalize
    if normalize_vectors:
        X_proc = _l2_normalize(X, axis=1)
        query_proc = _l2_normalize(query_vec, axis=1)
        if existing_vectors is not None:
            existing_proc = _l2_normalize(existing_vectors, axis=1)
        else:
            existing_proc = None

        # 이미 정규화했으므로 cosine similarity = dot product
        query_sim = (X_proc @ query_proc.T).reshape(-1)
        pairwise_sim_x = X_proc @ X_proc.T
    else:
        X_proc = X
        query_proc = query_vec
        existing_proc = existing_vectors

        query_sim = _cosine_similarity(X_proc, query_proc).reshape(-1)
        pairwise_sim_x = _cosine_similarity(X_proc)

    # candidate mask
    candidate_mask = np.ones(N, dtype=bool)
    if exclude_existing_from_candidates and len(existing_indices) > 0:
        candidate_mask[existing_indices] = False

    # initialize max similarity to selected/existing set
    max_sim_to_selected = np.full(N, -np.inf)

    has_any_seed = False

    # existing samples inside X
    if len(existing_indices) > 0:
        seed_sim_from_indices = pairwise_sim_x[:, existing_indices]
        if seed_sim_from_indices.ndim == 1:
            seed_sim_from_indices = seed_sim_from_indices.reshape(-1, 1)
        max_sim_to_selected = np.maximum(
            max_sim_to_selected,
            seed_sim_from_indices.max(axis=1)
        )
        has_any_seed = True

    # existing external vectors
    if existing_proc is not None and len(existing_proc) > 0:
        if normalize_vectors:
            sim_to_external = X_proc @ existing_proc.T
        else:
            sim_to_external = _cosine_similarity(X_proc, existing_proc)

        if sim_to_external.ndim == 1:
            sim_to_external = sim_to_external.reshape(-1, 1)

        max_sim_to_selected = np.maximum(
            max_sim_to_selected,
            sim_to_external.max(axis=1)
        )
        has_any_seed = True

    selected = []
    selected_scores = []

    # If no existing seed at all, choose first by highest relevance
    if not has_any_seed:
        masked_query_sim = np.where(candidate_mask, query_sim, -np.inf)
        first_idx = int(np.argmax(masked_query_sim))
        if np.isneginf(masked_query_sim[first_idx]):
            return ([], []) if return_scores else []
        selected.append(first_idx)
        selected_scores.append(float(query_sim[first_idx]))
        candidate_mask[first_idx] = False
        max_sim_to_selected = pairwise_sim_x[:, first_idx].copy()

    # Continue selecting until k new samples are chosen
    while len(selected) < k:
        if has_any_seed:
            # when seeded by existing set, use MMR directly from the beginning
            penalty = np.where(np.isfinite(max_sim_to_selected), max_sim_to_selected, 0.0)
        else:
            penalty = max_sim_to_selected

        scores = lambda_param * query_sim - (1 - lambda_param) * penalty
        scores[~candidate_mask] = -np.inf

        next_idx = int(np.argmax(scores))
        if np.isneginf(scores[next_idx]):
            break

        selected.append(next_idx)
        selected_scores.append(float(scores[next_idx]))
        candidate_mask[next_idx] = False

        max_sim_to_selected = np.maximum(max_sim_to_selected, pairwise_sim_x[:, next_idx])

    if return_scores:
        return selected, selected_scores
    return selected
