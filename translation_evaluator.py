import tkinter as tk
from tkinter import ttk, messagebox
import ssl
import certifi
from googletrans import Translator
from sacrebleu import sentence_bleu
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from collections import Counter

# Fix SSL certificate issue
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


class TranslationEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("Translation Quality Evaluator")
        self.root.geometry("1000x700")

        self.translator = Translator()
        self.languages = {'French': 'fr', 'German': 'de', 'Spanish': 'es'}
        self.tokenizer = Tokenizer13a()

        self.create_widgets()

    def create_widgets(self):
        """Create and arrange the UI components."""
        # Source text input
        tk.Label(self.root, text="Source Text (English):").pack(pady=(10, 0))
        self.source_text = tk.Text(self.root, height=10, width=80)
        self.source_text.pack(pady=(0, 10))

        # Language selection
        tk.Label(self.root, text="Target Language:").pack()
        self.language_var = tk.StringVar(value="French")
        language_menu = ttk.Combobox(self.root, textvariable=self.language_var, values=list(self.languages.keys()))
        language_menu.pack(pady=(0, 10))

        # User translation input
        tk.Label(self.root, text="Your Translation:").pack()
        self.user_translation = tk.Text(self.root, height=10, width=80)
        self.user_translation.pack(pady=(0, 10))

        # Evaluate button
        self.evaluate_button = tk.Button(self.root, text="Evaluate Translation", command=self.evaluate_translation)
        self.evaluate_button.pack(pady=(10, 20))

        # Result display
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=(10, 5))

        # Google translation display
        self.google_translation_label = tk.Label(self.root, text="Google Translation:", font=("Arial", 12))
        self.google_translation_label.pack(pady=(5, 0))
        self.google_translation_text = tk.Text(self.root, height=6, width=80, state="disabled")
        self.google_translation_text.pack(pady=(0, 10))

    def evaluate_translation(self):
        source_text = self.source_text.get("1.0", tk.END).strip()
        user_trans = self.user_translation.get("1.0", tk.END).strip()
        target_lang = self.languages[self.language_var.get()]

        try:
            # Get Google Translate version
            google_trans = self.translator.translate(source_text, dest=target_lang).text

            # Tokenize and normalize
            google_processed = self.tokenizer(google_trans)
            user_processed = self.tokenizer(user_trans)

            # Split into tokens for ROUGE calculation
            google_tokens = google_processed.split()
            user_tokens = user_processed.split()

            # Calculate scores
            bleu_score = round(sentence_bleu(user_processed, [google_processed]).score, 1)
            rouge_score = self.compute_rouge(google_tokens, user_tokens)
            combined_score = self.calculate_combined_score(bleu_score, rouge_score)

            # Display results with new combined score
            self.display_results(bleu_score, rouge_score, combined_score, google_trans)

        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

    def compute_rouge(self, ref_tokens, cand_tokens, n=1):
        """Calculate ROUGE-N F1 score between 0-100"""
        if not ref_tokens or not cand_tokens:
            return 0.0

        # Create n-gram counters
        ref_counts = Counter(zip(*[ref_tokens[i:] for i in range(n)])) if n > 1 else Counter(ref_tokens)
        cand_counts = Counter(zip(*[cand_tokens[i:] for i in range(n)])) if n > 1 else Counter(cand_tokens)

        # Calculate overlap
        overlap = sum((cand_counts & ref_counts).values())

        # Calculate precision and recall
        precision = overlap / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

        # Calculate F1 score
        if (precision + recall) == 0:
            return 0.0
        return round(2 * (precision * recall) / (precision + recall) * 100, 1)

    def calculate_combined_score(self, bleu, rouge):
        """Calculate harmonic mean of BLEU and ROUGE scores"""
        if (bleu + rouge) == 0:
            return 0.0
        return round(2 * (bleu * rouge) / (bleu + rouge), 1)

    def display_results(self, bleu, rouge, combined, google_trans):
        category, color = self.get_score_category(combined)
        result_text = (
            f"BLEU: {bleu}/100 | ROUGE-1: {rouge}/100\n"
            f"Combined Score: {combined}/100 - {category}"
        )

        self.result_label.config(text=result_text, foreground=color)
        self.show_google_translation(google_trans)

    def get_score_category(self, score):
        if score < 10:
            return ('Almost useless', 'red')
        elif 10 <= score < 20:
            return ('Hard to get the gist', 'orange')
        elif 20 <= score < 30:
            return ('Gist clear but significant errors', 'yellow')
        elif 30 <= score < 40:
            return ('Understandable to good', 'light green')
        elif 40 <= score < 50:
            return ('Quality translation', 'green')
        elif 50 <= score < 60:
            return ('High quality and fluent', 'dark green')
        else:
            return ('Highest quality', 'blue')

    def show_google_translation(self, translation):
        """Display Google's translation in the text box."""
        self.google_translation_text.config(state="normal")
        self.google_translation_text.delete("1.0", tk.END)
        self.google_translation_text.insert(tk.END, translation)
        self.google_translation_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationEvaluator(root)
    root.mainloop()