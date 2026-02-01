# translate.py
# This file handles translation. 
# I’m using googletrans since it’s easy to call and works with many Indian languages.
# The idea is simple: take the English sentence, send it to the translator,
# and print whatever we get back in a clean format.

from googletrans import Translator
import threading

# Create translator only once so it doesn’t re-load every time
_translator = Translator()

# Languages I want the sentence to appear in
LANG_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Gujarati": "gu",
}


def translate_sentence(text: str):
    results = {}

    # Always include English
    results["English"] = text

    if not GOOGLETRANS_AVAILABLE:
        return results

    LANG_CODES = {
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Kannada": "kn",
        "Gujarati": "gu",
    }

    for lang_name, code in LANG_CODES.items():
        try:
            translated = _translator.translate(text, src="en", dest=code)
            results[lang_name] = translated.text
        except Exception:
            results[lang_name] = "[Translation unavailable]"

    return results



def print_translations_to_console(sentence: str):
    """
    Prints all translations to the terminal.
    Added extra spacing because Indian scripts can look cramped.
    """
    translations = translate_sentence(sentence)

    print("\n" + "=" * 60)
    print("TRANSLATIONS")
    print("=" * 60 + "\n")

    # Order in which I want them to show
    order = ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Gujarati"]

    for lang in order:
        if lang in translations:
            print(f"{lang}:")
            print(f"  {translations[lang]}\n")   # newline makes it readable

    print("=" * 60 + "\n")


def translate_in_background(sentence: str):
    """
    Run translations in a separate thread.
    This way the main UI doesn't freeze while printing.
    """
    thread = threading.Thread(
        target=print_translations_to_console,
        args=(sentence,),
        daemon=True
    )
    thread.start()
