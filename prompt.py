FILTER_PROMPT = """ 
You are evaluating whether a given book review is meaningful based on the book description. A meaningful review should be relevant to the book's content, provide insights, and not be generic, vague, or off-topic.  

You will be given:  
1. **Book Description** – A brief summary of the book.  
2. **Review** – A review written about the book.  

Your task:  
- Output **"yes"** if the review is meaningful.  
- Output **"no"** if the review is not meaningful.  
- Provide **only** "yes" or "no" as your output.  

**Example 1 (Meaningful Review – Output: "yes")**  
**Book Description:** This is a historical fiction novel set in 18th-century France, following the life of a young noblewoman navigating political intrigue and personal betrayal.  
**Review:** "The book vividly captures the essence of 18th-century France, and the protagonist's struggles felt very real. The political drama was engaging, and the character development was deep and compelling."  
**Output:** yes  

**Example 2 (Not Meaningful Review – Output: "no")**  
**Book Description:** A science fiction novel about a team of astronauts discovering an alien civilization on a distant planet, exploring themes of first contact and artificial intelligence.  
**Review:** "This book was okay. Some parts were interesting, but I got bored at times. Not my favorite."  
**Output:** no  

**Input:**  
**Book Description:** {book_description}  
**Review:** {book_review}
**Output:**
"""

SUMMARY_PROMPT = """ 
You are evaluating multiple book reviews based on a given book description. Your task is to summarize the overall sentiment and key points from the reviews.  

You will be given:  
1. **Book Description** – A brief summary of the book.  
2. **Reviews** – A list of reviews written about the book.  

Your task:  
- Analyze the reviews and identify common themes, praises, and criticisms.  
- Provide a concise summary that captures the overall sentiment (e.g., positive, mixed, negative) and key insights from the reviews.  
- Ensure the summary remains relevant to the book’s content.  

**Example**  

**Book Description:** A historical fiction novel set in 18th-century France, following the life of a young noblewoman navigating political intrigue and personal betrayal.  

**Reviews:**  
1. "The book vividly captures the essence of 18th-century France, and the protagonist's struggles felt very real. The political drama was engaging."  
2. "Some sections were a bit slow, but overall, the historical details were fascinating, and the characters were well-developed."  
3. "I loved the intrigue and the emotional depth of the story. The writing was immersive, though some plot points felt predictable."  
4. "Too much focus on politics for my taste, but fans of historical fiction will likely enjoy it."  

**Summary:**  
The book is well-received for its rich historical details, strong character development, and engaging political intrigue. Readers appreciate the immersive storytelling and emotional depth, though some find certain sections slow or predictable. While most enjoy the political drama, a few feel it dominates the story too much. Overall, the reception is positive, especially among historical fiction fans.  

**Input:**  
**Book Description:** {book_description}  
**Reviews:**  
{reviews}
**Summary:** 
"""

TRANSLATE_PROMPT = """
You are translating a book summary into the specified target language.  

You will be given:  
1. **Summary** – A brief summary of a book.  
2. **Target Language** – The language into which the summary should be translated.  

Your task:  
- Translate the summary accurately while maintaining its meaning and readability.  
- Use natural and fluent phrasing in the target language.  

**Input:**  
**Summary:** {book_summary}  
**Target Language:** {language}
**Output:**
"""

DIGEST_PROMPT = """
You are summarizing a book review based on the book description. The summary should retain the key points while removing unnecessary details.  

You will be given:  
1. **Book Description** – A brief summary of the book.  
2. **Review** – A review written about the book.  

Your task:  
- Summarize the review in a concise yet informative manner.  
- Retain relevant details while removing generic or repetitive information.  
- Ensure the summary aligns with the book's content and themes.  

**Example**  
**Book Description:** A mystery novel set in a small coastal town, where a journalist investigates a decades-old unsolved murder, uncovering dark secrets about the town’s past.  
**Review:** "I really enjoyed this book! The small-town setting was atmospheric, and the mystery kept me guessing until the very end. The journalist protagonist was well-developed, and I liked how the story unfolded with unexpected twists. However, I felt the pacing was a bit slow in the middle."  
**Summary:** A gripping mystery with an immersive small-town setting and strong character development. The plot is full of unexpected twists, though the pacing slows in the middle.  

**Input:**  
**Book Description:** {book_description}  
**Review:** {review}
**Summary:**
"""
