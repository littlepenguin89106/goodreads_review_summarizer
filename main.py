import pandas as pd
from llm_client import build_llm_client
from prompt import FILTER_PROMPT, TRANSLATE_PROMPT, DIGEST_PROMPT
import os
from tqdm import tqdm
from utils import *
from book_platform import get_platform
import asyncio
from dotenv import load_dotenv
from omegaconf import OmegaConf
from collections import defaultdict
from functools import partial
from tqdm.contrib.logging import logging_redirect_tqdm

DEFAULT_LANGUAGE = "English"


async def filter_reviews(cfg, llm_client, book_info, review_df):
    async def filter_single_review(review):
        prompt = FILTER_PROMPT.format(
            book_description=book_info["description"], book_review=review.review
        )
        response = await llm_client.chat(prompt)

        return review if "yes" in response.lower() else None

    async def compute():
        logger.info("Filtering reviews...")

        results = await async_map_with_tqdm(
            filter_single_review, review_df.itertuples(), "Filter"
        )
        return pd.DataFrame(filter(None, results))

    cache_path = os.path.join(cfg.tmp_dir, "filtered.csv")

    return await async_use_cache_or_compute(cfg, cache_path, load_df, compute, save_df)


async def digest_reviews(cfg, llm_client, book_info, review_df):
    async def digest_single_review(review):
        prompt = DIGEST_PROMPT.format(
            book_title=book_info["title"],
            book_description=book_info["description"],
            review=review.review,
        )
        digest = await llm_client.chat(prompt)
        return digest

    async def compute():
        logger.info("Digesting reviews...")

        digested_list = await async_map_with_tqdm(
            digest_single_review, review_df.itertuples(), "Digest"
        )
        digested_df = review_df.copy()
        digested_df["review"] = digested_list
        return digested_df

    cache_path = os.path.join(cfg.tmp_dir, "digested.csv")
    return await async_use_cache_or_compute(cfg, cache_path, load_df, compute, save_df)


async def generate_summary(cfg, llm_client, book_info, review_list):
    async def compute():
        logger.info("Generating summary...")

        if cfg.clustering:
            cluster_labels = await cluster_reviews(cfg, llm_client, review_list)
            clusters = defaultdict(list)
            for review, label in zip(review_list, cluster_labels):
                clusters[label].append(review)
            clusters = dict(sorted(clusters.items(), key=lambda x: x[0]))

            sum_func = partial(summarize_group, llm_client, book_info)
            cluster_summaries = await async_map_with_tqdm(
                sum_func, clusters.values(), "Generating group summary"
            )
            total_summary = await sum_func(cluster_summaries)
            # Note: samples without labels are treated as group0
            summary = "\n".join(
                f"Group {i}: {s}" for i, s in enumerate(cluster_summaries)
            )
            summary += f"Total summary: {total_summary}"

        else:
            summary = await summarize_group(llm_client, book_info, review_list)

        return summary

    cache_path = os.path.join(get_save_dir(cfg), "summary.txt")
    return await async_use_cache_or_compute(
        cfg, cache_path, load_text, compute, save_text
    )


async def translate_summary(cfg, llm_client, summary):
    async def compute():
        logger.info("Translating summary...")
        if cfg.language == DEFAULT_LANGUAGE:
            return summary
        else:
            prompt = TRANSLATE_PROMPT.format(
                language=cfg.language, book_summary=summary
            )
            tranlated = await llm_client.chat(prompt)

        return tranlated

    cache_path = os.path.join(get_save_dir(cfg), "translated.txt")
    return await async_use_cache_or_compute(
        cfg, cache_path, load_text, compute, save_text
    )


async def summary_pipeline(cfg, llm_client, book_info, reviews):
    summary = await generate_summary(cfg, llm_client, book_info, reviews)
    tranlated_summary = await translate_summary(cfg, llm_client, summary)
    save_text(
        tranlated_summary,
        os.path.join(get_save_dir(cfg, "result"), f"summarized_result.txt"),
    )


async def main(cfg):

    with open(cfg.books, "r") as f:
        book_infos = f.read().splitlines()

    llm_client = build_llm_client(cfg.client)
    for info in tqdm(book_infos, desc="Books"):
        platform, book_dir, book_id = info.split(",")
        cfg.book_dir = os.path.join(cfg.output_dir, book_dir)
        make_dir_if_not_exists(cfg.book_dir)
        if cfg.save_tmp:
            cfg.tmp_dir = os.path.join(cfg.book_dir, "tmp")
            make_dir_if_not_exists(cfg.tmp_dir)

        with get_platform(cfg, platform) as platform:
            book_info = platform.get_book_info(book_id)
            reviews = platform.get_reviews(book_id)

        filtered_reviews = await filter_reviews(cfg, llm_client, book_info, reviews)
        digested_reviews = await digest_reviews(
            cfg, llm_client, book_info, filtered_reviews
        )

        if not digested_reviews.empty:
            if cfg.summary_mode == "rating":
                rating_max = digested_reviews["rating"].max()
                for rating in range(rating_max, -1, -1):
                    cfg.rating_dir = os.path.join(cfg.book_dir, f"rating_{rating}")
                    make_dir_if_not_exists(cfg.rating_dir)
                    if cfg.save_tmp:
                        cfg.rating_tmp_dir = os.path.join(
                            cfg.tmp_dir, f"rating_{rating}"
                        )
                        make_dir_if_not_exists(cfg.rating_tmp_dir)
                    select_reviews = digested_reviews[
                        (digested_reviews["rating"] >= rating)
                        & (digested_reviews["rating"] < rating + 1)
                    ]
                    if not select_reviews.empty:
                        select_list = select_reviews["review"].to_list()
                        await summary_pipeline(cfg, llm_client, book_info, select_list)
                    else:
                        logger.warning(
                            f"No insightful reviews found for rating: {rating}"
                        )
            else:
                digested_list = digested_reviews["review"].to_list()
                await summary_pipeline(cfg, llm_client, book_info, digested_list)
        else:
            logger.warning("No insightful reviews found for this book")

    logger.info("Done!")


def parse_config():
    load_dotenv()
    cfg = OmegaConf.load("configs/main.yaml")
    cfg.client = OmegaConf.load(os.path.join(f"configs/clients/{cfg.client}"))
    cfg.platform = OmegaConf.load(os.path.join(f"configs/{cfg.platform}"))

    return cfg


if __name__ == "__main__":
    cfg = parse_config()
    # fix new line promblem
    with logging_redirect_tqdm():
        asyncio.run(main(cfg))
