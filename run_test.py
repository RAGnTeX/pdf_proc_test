import os

from src.ragntex_processing import (
  extract_pdf_ragntex
)

def main():
  '''Main entry point for the script.'''

  # Get the papers to process
  papers_dir = 'papers'
  papers     = [f for f in os.listdir(papers_dir) if os.path.isdir(os.path.join(papers_dir, f))]

  # Process papers
  dataset_dir = 'dataset'
  for article in papers:
    article_dir = os.path.join(dataset_dir, article)
    os.makedirs(article_dir,  exist_ok=True)

    # Get the PDF file
    pdf_path = os.path.join(papers_dir, article, 'article.pdf')

    # Use RAGnTeX processing
    extract_pdf_ragntex(pdf_path, article_dir)



if __name__ == '__main__':
    main()
