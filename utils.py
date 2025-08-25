import json
import csv
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import os
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import umap
import matplotlib.pyplot as plt
from prompt import SUMMARY_PROMPT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger()


def get_save_dir(cfg, save_type="tmp"):
    if save_type == "tmp":
        return cfg.tmp_dir if cfg.summary_mode == "total" else cfg.rating_tmp_dir
    else:
        return cfg.book_dir if cfg.summary_mode == "total" else cfg.rating_dir


async def async_map_with_tqdm(func, iterable, desc=""):
    return await tqdm_asyncio.gather(*(func(item) for item in iterable), desc=desc)


def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def try_load_cache(cfg, loader, cache_path):
    if cfg["use_tmp"]:
        try:
            return loader(cache_path)
        except:
            logger.warning(f"Failed to load tmp file {cache_path}. Reprocessing.")
    return None


def maybe_save_cache(cfg, saver, result, cache_path):
    if cfg["save_tmp"]:
        if saver is not None:
            try:
                saver(result, cache_path)
            except Exception as e:
                logger.error(f"Failed to save tmp file {cache_path}")
                raise e


def sync_use_cache_or_compute(
    cfg,
    cache_path,
    loader,
    compute,
    saver,
):

    cached = try_load_cache(cfg, loader, cache_path)
    if cached is not None:
        return cached

    result = compute()
    maybe_save_cache(cfg, saver, result, cache_path)

    return result


async def async_use_cache_or_compute(
    cfg,
    cache_path,
    loader,
    compute,
    saver,
):

    cached = try_load_cache(cfg, loader, cache_path)
    if cached is not None:
        return cached

    result = await compute()
    maybe_save_cache(cfg, saver, result, cache_path)

    return result


def save_text(text, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return f.read()


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(arr, file_path):
    np.save(file_path, arr)


def load_npy(file_path):
    np.load(file_path)


def save_df(df, output_file):
    df.to_csv(
        output_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        lineterminator="\n",
    )


def load_df(input_file):
    return pd.read_csv(
        input_file, quoting=csv.QUOTE_MINIMAL, escapechar="\\", lineterminator="\n"
    )


def join_reviews(reviews):
    return "\n".join(f"{i}: {r}" for i, r in enumerate(reviews, 1))


async def summarize_group(llm_client, book_info, review_list):
    review_str = join_reviews(review_list)
    prompt = SUMMARY_PROMPT.format(
        book_title=book_info["title"],
        book_description=book_info["description"],
        reviews=review_str,
    )
    return await llm_client.chat(prompt)


async def embed_reviews(cfg, llm_client, review_list):
    async def compute():
        total_len = len(review_list)
        batch_size = cfg.client.embed_batch_size

        batch_list = []
        for i in range(0, total_len, batch_size):
            batch = review_list[i : i + batch_size]
            batch_list.append(batch)

        embeddings_list = await async_map_with_tqdm(
            llm_client.embed, batch_list, "Embed"
        )
        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings

    cache_path = os.path.join(get_save_dir(cfg), "embedding.npy")
    return await async_use_cache_or_compute(
        cfg, cache_path, load_npy, compute, save_npy
    )


def visualize_umap(cfg, embeddings, labels):

    reducer = umap.UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
        n_components=2,
        metric=cfg.umap.metric,
        random_state=cfg.seed,
    )
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    labels = np.asarray(labels)
    mask_noise = labels == -1
    if mask_noise.any():
        plt.scatter(
            coords[mask_noise, 0],
            coords[mask_noise, 1],
            s=10,
            alpha=0.4,
            label="cluster 0 (single)",
            c="lightgray",
        )
    for lab in np.unique(labels):
        if lab == -1:
            continue
        m = labels == lab
        plt.scatter(
            coords[m, 0], coords[m, 1], s=10, alpha=0.8, label=f"cluster {lab+1}"
        )
    plt.legend(markerscale=2, fontsize=8, loc="best")

    plt.title(f"UMAP visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(get_save_dir(cfg, "result"), f"cluster.png"), dpi=150)
    return coords


async def cluster_reviews(cfg, llm_client, review_list):
    embeddings = await embed_reviews(cfg, llm_client, review_list)

    def compute():

        # dimension reduction
        min_components = min(embeddings.shape[0], embeddings.shape[1])
        n_components = cfg.pca.n_components
        if n_components > min_components:
            n_components = min_components
            logger.warning(
                " must be between 0 and min(n_samples, n_features) with svd_solver='full', force set to reasonable value"
            )
        pca = PCA(n_components=n_components, random_state=cfg.seed)
        X_reduced = pca.fit_transform(embeddings)

        # clustering
        clusterer = HDBSCAN(
            min_cluster_size=cfg.hdbscan.min_cluster_size,
            min_samples=cfg.hdbscan.min_samples,
            metric=cfg.hdbscan.metric,
        )
        labels = clusterer.fit_predict(X_reduced)
        return labels

    cache_path = os.path.join(get_save_dir(cfg), "labels.npy")
    labels = sync_use_cache_or_compute(cfg, cache_path, load_npy, compute, save_npy)

    if cfg.viz_clusters:
        visualize_umap(cfg, embeddings, labels)

    return labels
