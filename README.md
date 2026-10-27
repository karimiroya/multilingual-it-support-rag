# Enterprise AI Support Copilot

I built this project to see how well a small, fully local RAG system can handle IT support questions, the kind employees send to a help desk every day ("my VPN doesn't connect", "Outlook isn't sending my emails"). Since I live in Italy, I wanted it to work in both Italian and English.

You write your problem in your own words, and the app:

1. guesses the category of the problem with a fine-tuned XLM-RoBERTa classifier,
2. searches a small knowledge base of Microsoft troubleshooting articles,
3. asks a local LLM (Qwen2.5 through Ollama) to write step-by-step instructions, citing the sources it used.

![Italian Wi-Fi question](screenshots/wifi_italian.png)

## How it works

```
question (IT or EN)
   -> XLM-R classifier (18 IT categories)
   -> hybrid search: BM25 + multilingual embeddings in FAISS, combined with Reciprocal Rank Fusion
   -> Qwen2.5-7B writes the answer: problem, possible cause, steps, sources
```

A few details:

- The classifier was fine-tuned on the [ticket_classification_IT_EN](https://huggingface.co/datasets/Noise144/ticket_classification_IT_EN) dataset.
- For embeddings I used `paraphrase-multilingual-MiniLM-L12-v2`, so an Italian question can match English documents.
- BM25 is my own small implementation. It is good with exact terms like "error 800", while embeddings are better with paraphrases. Combining them worked better than either one alone (see below).
- If the classifier is fairly sure (60% or more), the search is limited to that topic's documents. If it is sure the problem is about something the knowledge base doesn't cover, like a keyboard or a monitor, the app says so and suggests contacting IT instead of making something up.
- The knowledge base has 10 articles from Microsoft Learn and Microsoft Support (Wi-Fi, VPN, DNS, DHCP, email and general network problems), split into 164 chunks.

## How well does it work?

I wrote 42 test questions: 21 pairs in Italian and English. Most are specific ("VPN error 809"), some are vague ("my email is not working") and six are about things the system shouldn't answer (keyboard, printer, monitor). For each question I marked which chunks contain the answer.

Retrieval, on the 36 questions that should be answered:

| Method | Hit@1 | Hit@3 | MRR@10 |
|---|---|---|---|
| BM25 only | 38.9% | 61.1% | 0.510 |
| Embeddings only | 41.7% | 63.9% | 0.537 |
| Hybrid | 50.0% | 63.9% | 0.592 |
| Hybrid + category filter | 50.0% | 69.4% | 0.604 |

What I took from this:

- Hybrid search puts the right chunk first in half of the questions, about 10 points better than either method alone.
- Italian is harder because the documents are in English. BM25 alone gets only 27.8% Hit@1 on Italian questions; with the multilingual embeddings it goes up to 44.4%.
- Specific questions work well (83% Hit@3). Vague ones don't, more on that below.

Other numbers:

- The classifier is right on 82% of the test questions. Most mistakes are DNS or DHCP questions labelled "IP Address", which is understandable, so I now send that category to the DHCP, DNS and network documents too.
- 4 of the 6 out-of-scope questions are refused, and no normal question was refused by mistake.
- I compared Qwen2.5 7B and 3B. The 7B cites a source in 64% of its steps, against 19% for the 3B, and follows the answer format better. It is slower (8 s vs 4 s per answer on my MacBook) but I think it's worth it, so it's the default.

The evaluation also caught a bug I hadn't noticed while testing by hand: my first language detector, based on a short keyword list, thought 11 of the 21 Italian questions were English, so those people got answers in English. I replaced it with a simple word-count approach and now all 42 are detected correctly.

## What changed along the way

My first version answered "La connessione Wi-Fi aziendale non funziona" with a very confident explanation about a Windows 10 client failing 802.1X authentication. I never mentioned Windows 10 or 802.1X. The model had taken the scenario from the only Wi-Fi document I had, an admin guide, and presented it as the user's problem.

Two things fixed most of it:

- adding user-level guides (for example "Fix Wi-Fi connection issues in Windows") next to the admin documentation,
- telling the model clearly that facts about the problem can only come from the user's message, and that steps for specific setups should be written as conditions ("if your network uses 802.1X...").

![English VPN question](screenshots/vpn_english.png)

## Running it

You need Python (I used 3.13) and [Ollama](https://ollama.com).

```bash
git clone https://github.com/karimiroya/multilingual-it-support-rag.git
cd multilingual-it-support-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:7b

python build_knowledge_base.py   # downloads the articles and builds the index
streamlit run app.py
```

The fine-tuned classifier is too big for GitHub, so it isn't in the repository. The app expects it in `final_xlmr_ticket_model/`, and you can train it again with `01_dataset_exploration.ipynb`.

To try the smaller model: `LLM_MODEL=qwen2.5:3b streamlit run app.py`

To run the evaluation:

```bash
python evaluate.py              # retrieval and classifier only
python evaluate.py --generate   # also generates and checks the 42 answers
```

The results go to `data/evaluation/results/`, and the test questions are in `data/evaluation/eval_questions_v2.csv`.

## Files

- `app.py`: the Streamlit app
- `rag_engine.py`: search, prompt and answer generation
- `ticket_classifier.py`: loads the classifier and predicts the category
- `build_knowledge_base.py`: downloads the articles, splits them into chunks and builds the index
- `evaluate.py`: the evaluation
- `01_dataset_exploration.ipynb`: dataset exploration and classifier fine-tuning
- `rag_pipeline.ipynb`: my first version of the RAG pipeline
- `data/evaluation/v1/`: my first, smaller evaluation

## What still doesn't work well

- Vague questions. For "the Wi-Fi doesn't work" the search returns very specific sections (like "Wi-Fi option is missing") instead of the basic first steps, and the model never asks a follow-up question even though the prompt tells it to.
- If the classifier isn't confident, an out-of-scope question still gets an answer. "The printer doesn't print" is classified as Keyboard with 35% confidence, so it isn't refused and the answer comes from unrelated documents.
- 42 questions is a small test set, and I labelled them myself.
- The documents are public Microsoft articles, not a real company knowledge base.
- The 7B model sometimes makes small grammar mistakes in Italian.

If I continue, I'd like to try query rewriting for vague questions, a minimum relevance score to detect off-topic questions, and a small REST API with logging so the system could be connected to a real ticketing tool.
