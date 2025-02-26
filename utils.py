import json
import csv
from types import SimpleNamespace
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import os
import logging

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
