import json
import csv


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


def save_text(text, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_df(df, output_file):
    df.to_csv(
        output_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        lineterminator="\n",
    )
