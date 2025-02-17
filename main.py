import httpx
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import async_playwright
from argparse import ArgumentParser
from llm import OllamaClient
from prompt import FILTER_PROMPT, SUMMARY_PROMPT, TRANSLATE_PROMPT, DIGESGT_PROMPT
import os
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import logging
from utils import *
import time
import asyncio

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
DEFAULT_LANGUAGE = "English"


# from: https://stackoverflow.com/a/79037206
def parse_review_response(response):
    data = response["data"]["getReviews"]["edges"]
    review_list = []
    for d in data:
        try:
            review_list.append(
                {
                    "id_user": d["node"]["creator"]["id"],
                    "is_author": d["node"]["creator"]["isAuthor"],
                    "follower_count": d["node"]["creator"]["followersCount"],
                    "name": d["node"]["creator"]["name"],
                    "review": BeautifulSoup(d["node"]["text"], "lxml").get_text(
                        separator="\n"
                    ),
                    "review_create_data": d["node"]["createdAt"],
                    "review_liked": d["node"]["likeCount"],
                    "rating": d["node"]["rating"],
                }
            )
        except:
            logger.error("fail to extract request data")
            logger.error(d)

    return review_list


def get_reviews(args, request, rating, interval=2):
    if args.use_tmp:
        try:
            review_df = load_df(os.path.join(args.book_dir, f"reviews_{rating}.csv"))
        except:
            logger.warning(f"Failed to load tmp files of get_reviews, reprocessing for rating: {rating}")
        else:
            return review_df

    logger.info(f"Getting reviews for rating: {rating}")
    payload = request.post_data_json
    payload["variables"]["filters"]["ratingMax"] = rating
    payload["variables"]["filters"]["ratingMin"] = rating
    payload["variables"]["pagination"]["limit"] = args.reviews_per_chunk

    total_review_list = []
    review_count = 0
    # To avoid sending a large number of unofficial API requests, only synchronous requests are used here.
    with httpx.Client() as http_client:
        while review_count < args.max_reviews:
            response = http_client.post(
                request.url, json=payload, headers=request.headers
            )
            data = response.json()
            review_list = parse_review_response(data)
            total_review_list.extend(review_list)
            review_count += len(review_list)

            nextPageToken = data["data"]["getReviews"]["pageInfo"]["nextPageToken"]
            if not nextPageToken:
                break
            payload["variables"]["pagination"]["after"] = nextPageToken
            time.sleep(interval)

    review_df = pd.DataFrame(total_review_list)

    if args.save_tmp:
        save_df(review_df, os.path.join(args.book_dir, f"reviews_{rating}.csv"))
    return review_df

class ReviewRequestInterceptor:
    def __init__(self):
        self.review_request = None

    def handle_request(self, request):
        if "graphql" not in request.url or request.method != "POST":
            return

        post_data = request.post_data_json
        if (
            post_data
            and "operationName" in post_data
            and post_data["operationName"] == "getReviews"
        ):
            self.review_request = request


async def get_page(args, book_page):
    if args.use_tmp:
        try:
            book_info = load_json(os.path.join(args.book_dir, "book_info.json"))
            review_request = load_request(os.path.join(args.book_dir, "review_request.json"))
        except:
            logger.warning("Failed to load tmp files of get_page, reprocessing")
        else:
            return book_info, review_request

    logger.info(f"Getting book information")
    url = f"https://www.goodreads.com/book/show/{book_page}"
    request_interceptor = ReviewRequestInterceptor()

    book_info = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        context.on("request", request_interceptor.handle_request)
        await page.goto(url)

        title_selector = "div.BookPageTitleSection__title h1.Text.Text__title1"
        await page.wait_for_selector(title_selector, timeout=10000)
        book_info["title"] = await page.locator(title_selector).text_content()

        summary_selector = "div.BookPageMetadataSection__description span.Formatted"
        await page.wait_for_selector(summary_selector, timeout=10000)
        book_info["description"] = await page.locator(summary_selector).text_content()

        await browser.close()

    review_request = request_interceptor.review_request
    if args.save_tmp:
        save_json(book_info, os.path.join(args.book_dir, "book_info.json"))
        save_request(review_request, os.path.join(args.book_dir, "review_request.json"))

    return book_info, review_request

async def async_map_with_tqdm(func, iterable, desc=''):
    return await tqdm_asyncio.gather(*(func(item) for item in iterable),desc=desc)

async def filter_reviews(args, model_client, book_info, review_df, rating):
    async def filter_single_review(review):
        prompt = FILTER_PROMPT.format(
            book_description=book_info["description"], book_review=review.review
        )
        response = await model_client.chat(prompt)

        return review if 'yes' in response.lower() else None

    if args.use_tmp:
        try:
            filter_review_df = load_df(os.path.join(args.book_dir, f"filtered_{rating}.csv"))
        except:
            logger.warning(f"Failed to load tmp files of filter_reviews, reprocessing for rating: {rating}")
        else:
            return filter_review_df

    logger.info("Filtering reviews...")
    results = await async_map_with_tqdm(filter_single_review, review_df.itertuples(), "Reviews")

    filter_review_df = pd.DataFrame(filter(None, results))
    if args.save_tmp:
        save_df(
            filter_review_df,
            os.path.join(args.book_dir, f"filtered_{rating}.csv"),
        )
    return filter_review_df

async def digest_reviews(args, model_client, book_info, review_df, rating):
    async def digest_single_review(review):
        prompt = DIGESGT_PROMPT.format(
            book_title=book_info["title"],
            book_description=book_info["description"],
            review=review.review,
        )
        digest = await model_client.chat(prompt)
        return digest

    if args.use_tmp:
        try:
            digested_list = load_json(os.path.join(args.book_dir, f"digested_{rating}.json"))
        except:
            logger.warning(f"Failed to load tmp files of digest_reviews, reprocessing for rating: {rating}")
        else:
            return digested_list

    logger.info("Digesting reviews...")
    digested_list = await async_map_with_tqdm(digest_single_review, review_df.itertuples(), "Digests")

    if args.save_tmp:
        save_json(
            digested_list,
            os.path.join(args.book_dir, f"digested_{rating}.json"),
        )
    return digested_list


async def generate_summary(args, model_client, book_info, review_list, rating):
    if args.use_tmp:
        try:
            response = load_text(os.path.join(args.book_dir, f"summary_{rating}.txt"))
        except:
            logger.warning(f"Failed to load tmp files of generate_summary, reprocessing for rating: {rating}")
        
        else:
            return response

    logger.info("Generating summary...")
    review_str = ""
    for idx, review in enumerate(review_list, 1):
        review_str += f"{idx}: {review}\n"

    prompt = SUMMARY_PROMPT.format(
        book_title=book_info["title"],
        book_description=book_info["description"],
        reviews=review_str,
    )
    response = await model_client.chat(prompt)
    if args.save_tmp:
        save_text(response, os.path.join(args.book_dir, f"summary_{rating}.txt"))
    return response


async def translate_summary(args, model_client, summary, rating):
    if args.use_tmp:
        try:
            tranlated = load_text(os.path.join(args.book_dir, f"translated_{rating}.txt"))
        except:
            logger.warning(f"Failed to load tmp files of translate_summary, reprocessing for rating: {rating}")
        else:
            return tranlated

    logger.info("Translating summary...")
    if args.language == DEFAULT_LANGUAGE:
        return summary
    else:
        prompt = TRANSLATE_PROMPT.format(language=args.language, book_summary=summary)
        tranlated = await model_client.chat(prompt)
        if args.save_tmp:
            save_text(
                tranlated,
                os.path.join(args.book_dir, f"translated_{rating}.txt"),
            )
    return tranlated


def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


async def main(args):

    with open(args.books, "r") as f:
        book_pages = f.read().splitlines()

    model_client = OllamaClient(model=args.model, max_requests=args.max_llm_requests)
    for page in tqdm(book_pages, desc="Books"):
        logging.info(f"Processing book: {page}")
        args.book_dir = os.path.join(args.output_dir, page)
        make_dir_if_not_exists(args.book_dir)

        book_info, review_request = await get_page(args, page)
        if review_request:
            for rating in range(5, 0, -1):
                review_df = get_reviews(args, review_request, rating)
                filtered_reviews = await filter_reviews(
                    args, model_client, book_info, review_df, rating
                )
                if len(filtered_reviews):
                    digested_reviews = await digest_reviews(
                        args, model_client, book_info, filtered_reviews, rating
                    )
                    summary = await generate_summary(
                        args, model_client, book_info, digested_reviews, rating
                    )
                    tranlated_summary = await translate_summary(
                        args, model_client, summary, rating
                    )
                    save_text(
                        tranlated_summary,
                        os.path.join(args.book_dir, f"result_{rating}.txt"),
                    )
                else:
                    logger.warning(f"No reviews found for rating: {rating}")
        else:
            logger.warning("No reviews found for this book")

    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--books",
        type=str,
        required=True,
        help="Path to the file containing the list of book pages",
    )
    parser.add_argument(
        "--reviews_per_chunk",
        type=int,
        default=100,
        help="Number of reviews to request per chunk.",
    )
    parser.add_argument(
        "--max_reviews",
        type=int,
        default=100,
        help="Maximum number of reviews to scrape per book for each star rating.",
    )
    parser.add_argument(
        "--max_llm_requests",
        type=int,
        default=1,
        help="Maximum number of requests to send to the LLM model at once",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2", help="Processing model"
    )
    parser.add_argument(
        "--language", type=str, default=DEFAULT_LANGUAGE, help="Output language"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The directory to save the output",
    )
    parser.add_argument(
        "--save_tmp", default=False, action="store_true", help="Save intermediate data"
    )
    parser.add_argument("--use_tmp", default=False, action="store_true", help="Use intermediate data")
    args = parser.parse_args()
    logger = logging.getLogger()
    asyncio.run(main(args))
