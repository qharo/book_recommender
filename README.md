# 📚 Book Recommender

https://github.com/user-attachments/assets/9d3ced06-ac5e-44a7-87d1-e2483d84cebe

Built with Streamlit and an embedding model, it allows users to find their next read by describing what they're looking for or by selecting books they've previously enjoyed. The system explains recommendations, displays book covers and summaries, and provides links to additional information. It's containerized with Docker and configured for deployment on Fly.io. The application uses semantic vector representations to find similar books and features an intuitive, responsive interface.

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/qharo/book-recommender.git
   cd book-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run book_recommender.py
   ```

## Docker Deployment

You can also run the application using Docker:

```bash
docker build -t book-recommender .
docker run -p 8501:8501 book-recommender
```
