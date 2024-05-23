import nltk
from nltk import sent_tokenize
import tkinter as tk
from tkinter.filedialog import askopenfilename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
import io
from pypdf import PdfReader, PdfWriter
import re

nltk.download('stopwords')
nltk.download('punkt')

class FileBrowserApp:
    def __init__(self, root):
        self.root = root
        self.writer = PdfWriter()
        self.reader = None
        self.f_path_write = ""
        self.summary = None
        self.setup_ui()
        
    def setup_ui(self):
        self.file_explorer = tk.Label(self.root, text="Explore files",
                                      font=("Verdana", 14, "bold"),
                                      width=100, height=4,
                                      fg="white", bg="gray")
        self.button = tk.Button(self.root, text="Browse Folder",
                                font=("Roboto", 14), command=self.browse)
        self.file_explorer.pack()
        self.button.pack(pady=10)
    
    def browse(self):
        self.summary = Summarizer("")
        f_path = askopenfilename(initialdir="/",
                                 title="Select File",
                                 filetypes=(("PDF files", "*.pdf"), ("All Files", "*.*")))
        if f_path:
            self.file_explorer.configure(text="File Opened: " + f_path)
            self.f_path_write = f"{f_path[:f_path.rindex('/')]}/{f_path[f_path.rindex('/')+1:-4]}_summarized.pdf"
            self.reader = PdfReader(f_path)
            print("Starting summation")
            self.summary.extract_text_from_all_pages(self.reader)
            print("Finishing summation")
            self.create_pdf()

    def create_pdf(self):
        packet = io.BytesIO()

        doc = SimpleDocTemplate(packet, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        original_text = "Original Text:\n\n" + "\n\n".join(self.summary.original_sentences)
        story.append(Paragraph(original_text, styles["Normal"]))

        self.summary.generate_summary()  # Call the modified generate_summary method
        summary_text = self.summary.summary
        story.append(Paragraph(summary_text, styles["Normal"]))
        
        doc.build(story)

        packet.seek(0)

        new_pdf = PdfReader(packet)
        new_page = new_pdf.pages[0]

        self.writer.add_page(new_page)

        with open(self.f_path_write, "wb") as output_pdf:
            self.writer.write(output_pdf)

class Summarizer:
    def __init__(self, text):
        self.original_sentences = []  # Store original sentences
        self.text = text
        self.similarity_mat = None
        self.sentences = None
        self.sent_amount = 0
        self.summary = ""

    def extract_text_from_all_pages(self, reader):
        print("Starting reading text")
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() + ""
        print("Extracted raw text:", all_text[:500])  # Print the first 500 characters for debugging

        sentences = nltk.sent_tokenize(all_text)

        sentences = [sentence.strip() + '.' if not sentence.strip().endswith('.') else sentence.strip() for sentence in sentences if sentence.strip()]

        filtered_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 3 and len(sentence) < 300:
                if not re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$', sentence) and \
                   not re.search(r'\b(?:Abstract|Introduction|Conclusion|References|Figure|Table)\b', sentence):
                    filtered_sentences.append(sentence)

        self.original_sentences = filtered_sentences  # Save filtered sentences
        print("Filtered sentences:", self.original_sentences[:15])  # Print the first 5 filtered sentences for debugging

        self.text = [' '.join(self.remove_stopwords(sentence)) for sentence in self.original_sentences]  # Processed text for summarization
        print("Removed stopwords and filtered sentences:", self.text[:5])  # Print the first 5 processed sentences for debugging
        self.generate_similarity_matrix()


    def remove_stopwords(self, text):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        filtered_text = [word for word in text.split() if word.lower() not in stop_words]
        return filtered_text
    
    def generate_similarity_matrix(self):
        print("Generating similarity matrix from text")
        self.sentences = []
        for sentence in self.text:
            tokenized_sentence = nltk.sent_tokenize(sentence)
            self.sentences.extend(tokenized_sentence)
        print("Tokenized sentences:", len(self.sentences))

        vectorizer = TfidfVectorizer(norm='l2', smooth_idf=False)
        if not self.sentences:
            print("No sentences to vectorize.")
            return

        sentence_vectors = vectorizer.fit_transform(self.sentences)
        self.sent_amount = max(1, int(len(sentence_vectors.toarray()) * 0.35))
        print("Number of sentences to include in summary:", self.sent_amount)

        word_embeddings = sentence_vectors.toarray()
        self.similarity_mat = cosine_similarity(word_embeddings)
        print("Similarity matrix shape:", self.similarity_mat.shape)
        self.network_generator()


    def generate_summary(self):
        self.summary = ""
        sentences = nltk.sent_tokenize(self.summary)
        sentences_with_periods = [sentence.strip() + '.' if not sentence.strip().endswith('.') else sentence.strip() for sentence in sentences if sentence.strip()]
        self.summary = '.'.join(sentences_with_periods)


    def network_generator(self):
        if self.similarity_mat is None or self.similarity_mat.size == 0:
            print("Empty similarity matrix, cannot generate network.")
            return

        print("Generating network from similarity matrix")
        print("Length of original_sentences:", len(self.original_sentences))  # Debugging print
        G = nx.Graph()
        for i, sentence in enumerate(self.original_sentences):  # Add original sentences as nodes
            G.add_node(i, sentence=sentence)

        for i in range(len(self.sentences)):
            for j in range(len(self.sentences)):
                if i != j:
                    G.add_edge(i, j, weight=self.similarity_mat[i, j])

        page_rank_scores = nx.pagerank(G, weight='weight')
        num_sentences_in_summary = self.sent_amount
        sorted_sentences = sorted(page_rank_scores.items(), key=lambda item: item[1], reverse=True)
        top_sentences_indices = [item[0] for item in sorted_sentences[:num_sentences_in_summary]]
        print("Top sentence indices:", top_sentences_indices)  # Debugging print

        top_sentences_indices = [idx for idx in top_sentences_indices if idx < len(self.original_sentences)]

        top_sentences_indices.sort()
        top_sentences = [self.original_sentences[i] for i in top_sentences_indices]  # Use original sentences
        self.summary = ' '.join(top_sentences)
        print("Generated summary:", self.summary)


if __name__ == "__main__":
    root = tk.Tk()
    app = FileBrowserApp(root)
    root.mainloop()
