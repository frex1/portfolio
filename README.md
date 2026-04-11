# Portfolio

A robust, Python-based portfolio project featuring AI-powered data ingestion, semantic search, and gradio-based UI. Designed as a demonstration of full-stack data handling, retrieval, and AI-powered analytics or display.

---

## Features

- **Data Ingestion:** Scriptable ingestion pipeline (see `ingest.py`) for collecting and indexing diverse data sources (web, files, etc).
- **Semantic Search:** Utilizes state-of-the-art embedding models (sentence-transformers, FAISS) for fast, relevant info retrieval across the portfolio data.
- **Interactive Gradio UI:** User-friendly web interface for search, insights, visualization, and PDF report generation.
- **PDF Generation:** Export and share results/searches with formatted PDF reports.
- **Modern Python Stack:** Asynchronous HTTP (aiohttp), scraping (BeautifulSoup4), open API access, and environment configuration.

---

## Installation

### Requirements

- Python 3.12+
- See `pyproject.toml` for all dependencies (aiohttp, gradio, sentence-transformers, faiss, etc).

### Steps

1. **Clone the repository**
   ```sh
   git clone https://github.com/frex1/portfolio.git
   cd portfolio
   ```

2. **Install dependencies**
   ```sh
   pip install .
   ```

3. **Set environment variables**
   - Copy `.env.example` to `.env` and add any API keys or configuration required.

---

## Usage

- **Ingest data**  
  Run the data ingestion script to collect/update portfolio content:
  ```sh
  python ingest.py
  ```

- **Start the application**
  ```sh
  python app.py
  ```
  This launches the Gradio-based web app at a local URL (note output in terminal).

---

## Project Structure

```
.
├── app.py         # Main Gradio interface for portfolio search/UI
├── ingest.py      # Collect and index data sources
├── pyproject.toml # Project config & dependencies
├── .env           # Environment variables (after you create it)
└── ...
```

---

## Customization

- **Data sources:** Extend or replace data parsers in `ingest.py`.
- **Model selection:** Swap or tune embedding models for better domain relevance.
- **UI mods:** Adjust `app.py` for custom search, visualization or export features.

---

## License

MIT

---

## Issues & Support

- Report bugs, request features, or contribute via https://github.com/frex1/portfolio/issues

---

## Author

- https://github.com/frex1