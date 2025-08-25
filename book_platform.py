from bs4 import BeautifulSoup
import copy
import httpx
import os
from utils import *
import time
import re

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
}


def get_platform(cfg, platform):
    mapping = {
        "goodreads": Goodreads,
        "storygraph": StoryGraph,
    }
    try:
        return mapping[platform](cfg)
    except:
        raise ValueError(f"Do not support such platform: {platform}")


class BookPlatform:
    def __init__(self, **kwargs):
        # To avoid sending a large number of unofficial API requests, only synchronous requests are used here.
        if "headers" not in kwargs:
            kwargs["headers"] = DEFAULT_HEADERS
        self.client = httpx.Client(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()

    def get_book_info(self, book_id):
        cache_path = os.path.join(self.cfg.tmp_dir, "book_info.json")

        def compute():
            logger.info("getting book info")
            return self._get_book_info(book_id)

        return sync_use_cache_or_compute(
            self.cfg, cache_path, load_json, compute, save_json
        )

    def _get_book_info(self, book_id):
        pass

    def get_reviews(self, book_id):
        def compute():
            logger.info("get reviews")
            return self._get_reviews(book_id)

        cache_path = os.path.join(self.cfg.tmp_dir, "reviews.csv")
        return sync_use_cache_or_compute(
            self.cfg,
            cache_path,
            load_df,
            compute,
            save_df,
        )

    async def _get_reviews(self, book_id):
        pass


class Goodreads(BookPlatform):
    # reference: https://stackoverflow.com/a/79037206
    def __init__(self, cfg):
        self.cfg = cfg
        self.plat_cfg = cfg.platform.goodreads
        super().__init__()

    def _get_book_info(self, book_id):

        book_info = {}
        with httpx.Client() as http_client:
            response = http_client.get(f"https://www.goodreads.com/book/show/{book_id}")
            soup = BeautifulSoup(response.text, "lxml")
            book_info["title"] = (
                soup.select_one(".Text.Text__title1").get_text().strip()
            )
            book_info["description"] = (
                soup.select_one(
                    ".BookPageMetadataSection__description .TruncatedContent__text span"
                )
                .get_text()
                .strip()
            )
        logger.info(f"Now processing: {book_info['title']}")

        return book_info

    def get_work_id(self, book_id):
        def compute():
            response = self.client.get(
                f"https://www.goodreads.com/book/show/{book_id}/reviews"
            )
            soup = BeautifulSoup(response.text, "lxml")
            script_text = soup.find("script", id="__NEXT_DATA__").get_text()
            search_dict = json.loads(script_text)["props"]["pageProps"]["apolloState"]
            work_id = None
            for k in search_dict:
                if k.startswith("Work:"):
                    work_id = search_dict[k]["id"]

            assert work_id is not None, "Can not get work_id"

        cache_path = os.path.join(cfg.tmp_dir, "work_id.txt")
        return sync_use_cache_or_compute(cfg, cache_path, load_text, compute, load_text)

    def get_rating_reviews(self, work_id, rating=None):
        payload = {
            "operationName": "getReviews",
            "variables": {
                "filters": {"resourceType": "WORK", "resourceId": work_id},
                "pagination": {},
            },
            "query": "query getReviews($filters: BookReviewsFilterInput!, $pagination: PaginationInput) {\n  getReviews(filters: $filters, pagination: $pagination) {\n    ...BookReviewsFragment\n    __typename\n  }\n}\n\nfragment BookReviewsFragment on BookReviewsConnection {\n  totalCount\n  edges {\n    node {\n      ...ReviewCardFragment\n      __typename\n    }\n    __typename\n  }\n  pageInfo {\n    prevPageToken\n    nextPageToken\n    __typename\n  }\n  __typename\n}\n\nfragment ReviewCardFragment on Review {\n  __typename\n  id\n  creator {\n    ...ReviewerProfileFragment\n    __typename\n  }\n  recommendFor\n  updatedAt\n  createdAt\n  spoilerStatus\n  lastRevisionAt\n  text\n  rating\n  shelving {\n    shelf {\n      name\n      displayName\n      editable\n      default\n      actionType\n      sortOrder\n      webUrl\n      __typename\n    }\n    taggings {\n      tag {\n        name\n        webUrl\n        __typename\n      }\n      __typename\n    }\n    webUrl\n    __typename\n  }\n  likeCount\n  viewerHasLiked\n  commentCount\n}\n\nfragment ReviewerProfileFragment on User {\n  id: legacyId\n  imageUrlSquare\n  isAuthor\n  ...SocialUserFragment\n  textReviewsCount\n  viewerRelationshipStatus {\n    isBlockedByViewer\n    __typename\n  }\n  name\n  webUrl\n  contributor {\n    id\n    works {\n      totalCount\n      __typename\n    }\n    __typename\n  }\n  __typename\n}\n\nfragment SocialUserFragment on User {\n  viewerRelationshipStatus {\n    isFollowing\n    isFriend\n    __typename\n  }\n  followersCount\n  __typename\n}",
        }

        if rating:
            payload["variables"]["filters"]["ratingMax"] = rating
            payload["variables"]["filters"]["ratingMin"] = rating
        payload["variables"]["pagination"]["limit"] = self.plat_cfg.reviews_per_request

        total_review_list = []
        review_count = 0
        max_reviews = self.plat_cfg.max_reviews
        headers = copy.copy(DEFAULT_HEADERS)
        headers["X-Api-Key"] = "da2-xpgsdydkbregjhpr6ejzqdhuwy"

        while review_count < max_reviews:
            response = self.client.post(
                "https://kxbwmqov6jgg3daaamb744ycu4.appsync-api.us-east-1.amazonaws.com/graphql",
                json=payload,
                headers=headers,
            )
            data = response.json()
            review_list = self.parse_review_response(data)
            remaining = max_reviews - review_count

            total_review_list.extend(review_list[:remaining])
            review_count += min(len(review_list), remaining)

            if review_count >= max_reviews:
                break

            nextPageToken = data["data"]["getReviews"]["pageInfo"]["nextPageToken"]
            if not nextPageToken:
                break
            payload["variables"]["pagination"]["after"] = nextPageToken
            time.sleep(self.plat_cfg.interval)

        return total_review_list

    def _get_reviews(self, book_id):
        work_id = self.get_work_id(book_id)
        if self.plat_cfg.limit_by == "rating":
            review_list = []
            for rating in range(5, 0, -1):
                review_list.extend(self.get_rating_reviews(work_id, rating))
        else:
            review_list = self.get_rating_reviews(work_id)
        reviews = pd.DataFrame(review_list)

        return reviews

    @staticmethod
    def parse_review_response(response):
        data = response["data"]["getReviews"]["edges"]
        reviews = []
        for d in data:
            try:
                reviews.append(
                    {
                        "id_user": d["node"]["creator"]["id"],
                        "is_author": d["node"]["creator"]["isAuthor"],
                        "follower_count": d["node"]["creator"]["followersCount"],
                        "name": d["node"]["creator"]["name"],
                        "review": BeautifulSoup(d["node"]["text"], "lxml")
                        .get_text(separator="\n")
                        .strip(),
                        "review_create_date": d["node"]["createdAt"],
                        "review_liked": d["node"]["likeCount"],
                        "rating": d["node"]["rating"],
                    }
                )
            except:
                logger.error(f"fail to extract request data: {d}")

        return reviews


class StoryGraph(BookPlatform):
    def __init__(self, cfg):
        self.cfg = cfg
        self.plat_cfg = self.cfg.platform.storygraph
        assert (
            self.plat_cfg.limit_by == "total"
        ), "The StoryGraph only support limit_by=total"
        # need larger timeout
        super().__init__(timeout=20)

    def get_description(self, soup):
        # unsure if the "read more" button will always be present
        if soup.select_one(".read-more-btn"):
            script_text = soup.select_one("main  div  script").get_text()
            match = re.search(r"\.html\('(.*)'\)", script_text, re.DOTALL)
            assert match is not None, "Failed to parse description"
            match_text = match.group(1).replace("\\/", "/")
            description = BeautifulSoup(match_text, "lxml").get_text(separator="\n")
            skip_len = len("Description\n")
            description = description[skip_len:]
        else:
            description = (
                soup.select_one(".book-pane.blurb-pane p.trix-content")
                .get_text()
                .strip()
            )
        return description

    def _get_book_info(self, book_id):
        book_info = {}
        response = self.client.get(f"https://app.thestorygraph.com/books/{book_id}")
        soup = BeautifulSoup(response.text, "lxml")
        book_info["title"] = (
            soup.select_one(".book-title-author-and-series h3").get_text().strip()
        )
        book_info["description"] = self.get_description(soup)

        return book_info

    def _get_reviews(self, book_id):
        book_url = f"https://app.thestorygraph.com/book_reviews/{book_id}"
        params = {"written_explanations_only": True}
        max_reviews = self.plat_cfg.max_reviews
        rating_re = re.compile(r"Book rating:\s*(\d+(?:\.\d+)?) out of 5")

        reviews = []
        review_count = 0
        page = 1

        while review_count < max_reviews:
            if page > 1:
                params["page"] = page
            resp = self.client.get(book_url, params=params)
            soup = BeautifulSoup(resp.text, "lxml")
            review_elements = soup.select(".review-panes .standard-pane")

            # no more reviews
            if not review_elements:
                break

            for ele in review_elements:
                review_ele = ele.select_one(".review-explanation")
                review_text = review_ele.get_text(strip=True)

                user_ele = ele.select_one(".inverse-link")
                user_id = user_ele.get_text(strip=True)

                rating_ele = ele.find(attrs={"aria-label": rating_re})
                if rating_ele:
                    label = rating_ele["aria-label"]
                    match = rating_re.search(label)
                    rating = float(match.group(1))
                else:
                    rating = None

                reviews.append(
                    {
                        "id_user": user_id,
                        "review": review_text,
                        "rating": rating,
                    }
                )
                review_count += 1

            page += 1
        return pd.DataFrame(reviews)


if __name__ == "__main__":
    from main import parse_config

    cfg = parse_config()
    with StoryGraph(cfg) as platform:
        pass
