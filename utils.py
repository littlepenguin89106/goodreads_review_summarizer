import json
import csv
from types import SimpleNamespace
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


async def async_map_with_tqdm(func, iterable, desc=""):
    return await tqdm_asyncio.gather(*(func(item) for item in iterable), desc=desc)


def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_request(request, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "url": request.url,
                "headers": request.headers,
                "post_data_json": request.post_data_json,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


def load_request(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        request_dict = json.load(f)
    return SimpleNamespace(**request_dict)


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
    return '\n'.join(f"{i}: {r}" for i, r in enumerate(reviews, 1))

async def summarize_group(llm_client, book_info, review_list):
    review_str = join_reviews(review_list)
    prompt = SUMMARY_PROMPT.format(
        book_title=book_info["title"],
        book_description=book_info["description"],
        reviews=review_str,
    )
    return await llm_client.chat(prompt)

async def embed_reviews(cfg, llm_client,review_list):
    if cfg.use_tmp:
        try:
            embeddings = np.load(os.path.join(cfg.rating_tmp_dir, f'embeddings.npy'))
        except:
            logger.warning(f'Failed to load embedding, reprocessing.')
        else:
            return embeddings
    total_len = len(review_list)
    batch_size = cfg.client.embed_batch_size

    batch_list = []
    for i in range(0, total_len, batch_size):
        batch = review_list[i:i + batch_size]
        batch_list.append(batch)

    embeddings_list = await async_map_with_tqdm(llm_client.embed, batch_list, 'Embed')
    embeddings = np.concatenate(embeddings_list, axis=0)

    if cfg.save_tmp:
        np.save(os.path.join(cfg.rating_tmp_dir, f'embeddings.npy'), embeddings)
    
    return embeddings

def visualize_umap(cfg, embeddings, labels):

    reducer = umap.UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        min_dist=cfg.umap.min_dist,
        n_components=2,
        metric=cfg.umap.metric,
        random_state=cfg.seed
    )
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    labels = np.asarray(labels)
    mask_noise = labels == -1
    if mask_noise.any():
        plt.scatter(coords[mask_noise, 0], coords[mask_noise, 1],
                    s=10, alpha=0.4, label="single", c="lightgray")
    for lab in np.unique(labels):
        if lab == -1: 
            continue
        m = labels == lab
        plt.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.8, label=f"cluster {lab}")
    plt.legend(markerscale=2, fontsize=8, loc="best")

    plt.title(f"UMAP visualization")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.rating_dir,f'cluster.png'), dpi=150)
    return coords

async def cluster_reviews(cfg, llm_client, review_list):
    embeddings = await embed_reviews(cfg,llm_client,review_list)

    if cfg.use_tmp:
        try:
            labels = np.load(os.path.join(cfg.rating_tmp_dir, f'labels.npy'))
        except:
            logger.warning('Failed to load cluserting labels, reprocessing.')
        else:
            return labels

    # dimension reduction
    pca = PCA(n_components=cfg.pca.n_components, random_state=cfg.seed)
    X_reduced = pca.fit_transform(embeddings)

    # clustering
    clusterer = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric= cfg.hdbscan.metric
    )
    labels = clusterer.fit_predict(X_reduced)

    if cfg.viz_clusters:
        visualize_umap(cfg,embeddings,labels)

    if cfg.save_tmp:
        np.save(os.path.join(cfg.rating_tmp_dir, f'labels.npy'), labels)

    return labels