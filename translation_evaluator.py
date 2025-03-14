import tkinter as tk
from tkinter import ttk, messagebox
import ssl
import certifi
from googletrans import Translator, LANGUAGES

from sacrebleu import sentence_bleu
import nltk
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

# SSL certificate
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Download NLTK punctuation rules with proper SSL context
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TranslationEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("Translation Quality Evaluator")
        self.root.geometry("800x300")

        self.translator = Translator()
        self.languages = {'French': 'fr', 'German': 'de', 'Spanish': 'es'}

        self.create_widgets()

    def create_widgets(self):
        # Source Text
        ttk.Label(self.root, text="English Source Text (max 2000 chars):").grid(row=0, column=0, padx=10, pady=5,
                                                                                sticky='w')
        self.source_text = tk.Text(self.root, height=8, width=60, wrap=tk.WORD)
        self.source_text.grid(row=1, column=0, padx=10, pady=5)

        # Translation Input
        ttk.Label(self.root, text="Your Translation (max 2000 chars):").grid(row=0, column=1, padx=10, pady=5,
                                                                             sticky='w')
        self.user_translation = tk.Text(self.root, height=8, width=60, wrap=tk.WORD)
        self.user_translation.grid(row=1, column=1, padx=10, pady=5)

        # Language Selection
        ttk.Label(self.root, text="Target Language:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.language_var = tk.StringVar()
        self.lang_combo = ttk.Combobox(self.root, textvariable=self.language_var,
                                       values=list(self.languages.keys()), state='readonly')
        self.lang_combo.grid(row=2, column=0, padx=10, pady=5, sticky='e')
        self.lang_combo.current(0)

        # Evaluate Button
        self.eval_btn = ttk.Button(self.root, text="Evaluate Translation", command=self.evaluate_translation)
        self.eval_btn.grid(row=2, column=1, padx=10, pady=5, sticky='w')

        # Results Display
        self.result_label = ttk.Label(self.root, text="", font=('Helvetica', 12))
        self.result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=20)

    def populate_score_explanation(self):
        explanation = """Score Guide:
- < 10: Almost useless (red)
- 10 - 19: Hard to get the gist (orange)
- 20 - 29: Gist clear but significant errors (yellow)
- 30 - 40: Understandable to good (light green)
- 40 - 50: Quality translation (green)
- 50 - 60: High quality and fluent (dark green)
- > 60: Highest quality (blue)"""

        self.score_explanation.config(state='normal')
        self.score_explanation.delete(1.0, tk.END)
        self.score_explanation.insert(tk.END, explanation)
        self.score_explanation.config(state='disabled')

    def evaluate_translation(self):
        source_text = self.source_text.get("1.0", tk.END).strip()
        user_trans = self.user_translation.get("1.0", tk.END).strip()
        target_lang = self.languages[self.language_var.get()]

        try:
            # Get translations
            google_trans = self.translator.translate(source_text, dest=target_lang).text

            # Tokenize with SacreBLEU's tokenizer
            tokenizer = Tokenizer13a()
            google_processed = tokenizer(google_trans)
            user_processed = tokenizer(user_trans)

            # Calculate BLEU
            score = round(sentence_bleu(user_processed, [google_processed]).score, 1)

            self.display_results(score, google_trans)

        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")

    def get_score_category(self, score):
        if score < 10:
            return ('Almost useless', 'red')
        elif 10 <= score < 20:
            return ('Hard to get the gist', 'orange')
        elif 20 <= score < 30:
            return ('The gist is clear, but has significant grammatical errors', 'yellow')
        elif 30 <= score < 40:
            return ('Understandable to good translations', 'light green')
        elif 40 <= score < 50:
            return ('Quality translation', 'green')
        elif 50 <= score < 60:
            return ('High quality, adequate, and fluent translation', 'dark green')
        else:
            return ('Highest quality', 'blue')

    def display_results(self, score, google_trans):
        category, color = self.get_score_category(score)
        result_text = f"Score: {score}/100 - {category}"

        self.result_label.config(text=result_text, foreground=color)

        # Show Google translation in a new window
        self.show_google_translation(google_trans)

    def show_google_translation(self, text):
        top = tk.Toplevel(self.root)
        top.title("Google Translation Reference")

        text_area = tk.Text(top, wrap=tk.WORD, width=80, height=15)
        text_area.insert(tk.END, text)
        text_area.config(state='disabled')
        text_area.pack(padx=10, pady=10)

        close_btn = ttk.Button(top, text="Close", command=top.destroy)
        close_btn.pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationEvaluator(root)
    root.mainloop()