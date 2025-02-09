# Goodreads Review Summarizer

## Overview

Goodreads Review Summarizer is a tool to collect and summarize book reviews from Goodreads. The goal of this project is to extract insights from multiple user reviews, providing a informative summary of a book's reception.

## Features

- Scrapes and summarizes book reviews from Goodreads
- Output the structured summaries by AI to help users make informed reading choices

## Installation

1. Ensure you have Python3 installed. This project was developed using Python 3.10.
2. Run `pip install -r requirements.txt`
3. Run `playwright install` for installing playwright dependencies.
4. Install and run [Ollama](https://ollama.com/). Pull a suitable model.

## Usage

1. Browse the Goodreads page of the book you're interested in and note its identifier.

For example, the Goodreads page for *Atomic Habits* is:
https://www.goodreads.com/book/show/40121378-atomic-habits

In this case, `40121378-atomic-habits` is the identifier you need.

2. Copy and paste the book identifier into a text file. This tool supports multiple books, so you can enter multiple identifiers, each on a new line.

Refer to `books.txt` for an example.

3. Execute `python main.py --books books.txt`, replace `books.txt` with your list of book IDs. For other settings, run `python main.py --help` for more details.

## Notices

- Obtaining Goodreads reviews is not through an officially public API. Please be mindful of the frequency of requests and the amount of data being retrieved.
- Summary is generated by LLM, please verify the generated content

## Planned Features

- [ ] frontend design
- [ ] backend service
- [ ] Optimize summary method
- [ ] Support other model API
- [ ] Change to asynchronous requests to improve speed
