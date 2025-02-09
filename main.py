import requests
from bs4 import BeautifulSoup
import pandas as pd
from playwright.sync_api import sync_playwright
from argparse import ArgumentParser
from llm import OllamaClient
from prompt import FILTER_PROMPT, SUMMARY_PROMPT, TRANSLATE_PROMPT, DIGESGT_PROMPT
import os
from tqdm import tqdm
import logging
from utils import save_request, save_text, save_df, save_json
import time

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


def get_reviews(args, request, rating):
    logger.info(f"Getting reviews for rating: {rating}")
    payload = request.post_data_json
    payload["variables"]["filters"]["ratingMax"] = rating
    payload["variables"]["filters"]["ratingMin"] = rating
    payload["variables"]["pagination"]["limit"] = args.num_request_chunk

    total_review_list = []
    review_count = 0
    while review_count < args.max_request_limit:
        response = requests.post(request.url, json=payload, headers=request.headers)
        data = response.json()
        review_list = parse_review_response(data)
        total_review_list.extend(review_list)
        review_count += len(review_list)

        nextPageToken = data["data"]["getReviews"]["pageInfo"]["nextPageToken"]
        if not nextPageToken:
            break
        payload["variables"]["pagination"]["after"] = nextPageToken
        time.sleep(3)

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


def get_page(args, book_page):
    logger.info(f"Getting book information")
    url = f"https://www.goodreads.com/book/show/{book_page}"
    request_interceptor = ReviewRequestInterceptor()

    book_info = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        context.on("request", request_interceptor.handle_request)
        page.goto(url)

        title_selector = "div.BookPageTitleSection__title h1.Text.Text__title1"
        page.wait_for_selector(title_selector, timeout=10000)
        book_info["title"] = page.locator(title_selector).text_content()

        summary_selector = "div.BookPageMetadataSection__description span.Formatted"
        page.wait_for_selector(summary_selector, timeout=10000)
        book_info["description"] = page.locator(summary_selector).text_content()

        browser.close()

    review_request = request_interceptor.review_request
    if args.save_tmp:
        save_json(book_info, os.path.join(args.book_dir, "book_info.json"))
        save_request(review_request, os.path.join(args.book_dir, "review_request.json"))

    return book_info, review_request


def filter_reviews(args, client, book_info, review_df, rating):
    filtered_reviews = []
    logger.info("Filtering reviews...")
    for review in tqdm(review_df.itertuples(), "Reviews"):
        prompt = FILTER_PROMPT.format(
            book_description=book_info["description"], book_review=review.review
        )
        response = client.generate(prompt)

        if "yes" in response.lower():
            filtered_reviews.append(review)

    filter_review_df = pd.DataFrame(filtered_reviews)
    if args.save_tmp:
        save_df(
            filter_review_df,
            os.path.join(args.book_dir, f"filtered_{rating}.csv"),
        )
    return filter_review_df


def digest_reviews(args, client, book_info, review_df, rating):
    logger.info("Digesting reviews...")
    digested_list = []
    for review in tqdm(review_df.itertuples()):
        prompt = DIGESGT_PROMPT.format(
            book_title=book_info["title"],
            book_description=book_info["description"],
            review=review.review,
        )
        digest = client.generate(prompt)
        digested_list.append(digest)

    if args.save_tmp:
        save_json(
            digested_list,
            os.path.join(args.book_dir, f"digested_{rating}.json"),
        )
    return digested_list


def generate_summary(args, client, book_info, review_list, rating):
    logger.info("Generating summary...")
    review_str = ""
    for idx, review in enumerate(review_list, 1):
        review_str += f"{idx}: {review}\n"

    prompt = SUMMARY_PROMPT.format(
        book_title=book_info["title"],
        book_description=book_info["description"],
        reviews=review_str,
    )
    response = client.generate(prompt)
    if args.save_tmp:
        save_text(response, os.path.join(args.book_dir, f"summary_{rating}.txt"))
    return response


def translate_summary(args, client, summary, rating):
    logger.info("Translating summary...")
    if args.language == DEFAULT_LANGUAGE:
        return summary
    else:
        prompt = TRANSLATE_PROMPT.format(language=args.language, review_summary=summary)
        tranlated = client.generate(prompt)
        if args.save_tmp:
            save_text(
                tranlated,
                os.path.join(args.book_dir, f"translated_{rating}.txt"),
            )
        return tranlated


def make_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(args):

    with open(args.books, "r") as f:
        book_list = f.read().splitlines()

    client = OllamaClient(model=args.model_name)
    for book_page in tqdm(book_list, desc="Books"):
        logging.info(f"Processing book: {book_page}")
        args.book_dir = os.path.join(args.output_dir, book_page)
        make_dir_if_not_exists(args.book_dir)

        book_info, review_request = get_page(args, book_page)
        if review_request:
            for rating in range(5, 0, -1):
                review_df = get_reviews(args, review_request, rating)
                filtered_reviews = filter_reviews(
                    args, client, book_info, review_df, rating
                )
                if len(filtered_reviews):
                    digested_reviews = digest_reviews(
                        args, client, book_info, filtered_reviews, rating
                    )
                    summary = generate_summary(
                        args, client, book_info, digested_reviews, rating
                    )
                    tranlated_summary = translate_summary(args, client, summary, rating)
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
        "--num_request_chunk",
        type=int,
        default=100,
        help="Number of reviews to request per chunk.",
    )
    parser.add_argument(
        "--max_request_limit",
        type=int,
        default=100,
        help="Maximum number of reviews to scrape per book for each star rating.",
    )
    parser.add_argument(
        "--model_name", type=str, default="llama3.2", help="Processing model name"
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
    args = parser.parse_args()
    logger = logging.getLogger()
    main(args)
