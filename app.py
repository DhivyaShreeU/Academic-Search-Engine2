from flask import Flask, render_template, request
import wikipedia

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        query = request.form["query"]

        documents = []
        urls = []

        # Search Wikipedia
        search_results = wikipedia.search(query, results=5)

        for topic in search_results:
            try:
                page = wikipedia.page(topic)

                documents.append(page.summary)
                urls.append(page.url)

            except:
                continue

        if len(documents) == 0:
            return render_template("index.html", query=query)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(documents + [query])

        query_vector = tfidf_matrix[-1]
        document_vectors = tfidf_matrix[:-1]

        # Cosine Similarity
        similarity_scores = cosine_similarity(query_vector, document_vectors)[0]

        # Rank documents
        ranked = sorted(
            list(enumerate(similarity_scores)),
            key=lambda x: x[1],
            reverse=True
        )

        # Top results for banner
        most_relevant_docs = []

        for doc_id, score in ranked[:3]:
            title = search_results[doc_id]
            url = urls[doc_id]

            most_relevant_docs.append(f"{title} → {url}")

        return render_template(
            "index.html",
            query=query,
            ranked=ranked,
            documents=documents,
            urls=urls,
            most_relevant_docs=most_relevant_docs
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)