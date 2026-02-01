# grammar.py
# This file converts the list of recognized gesture words
# into a more natural English sentence.

def clean_sentence(words):
    """
    Simple formatting:
    joins words with spaces, adds capital letter and a dot.
    Example:
        ["HELLO", "INDIAN", "SIGN", "LANGUAGE"]
        -> "Hello Indian Sign Language."
    """
    if not words:
        return ""
    s = " ".join(words)
    s = s.title()
    if not s.endswith("."):
        s += "."
    return s


def grammar_correction(words):
    """
    Check for known patterns and return a nicer sentence.
    If no rule matches, fall back to clean_sentence().
    """
    # Work in uppercase for pattern matching
    w = [x.upper() for x in words]

    # --------- Custom rules for common phrases ---------

    # Intro about the system
    if w == ["HELLO", "INDIAN", "SIGN", "LANGUAGE"]:
        return "Hello, this is Indian Sign Language."

    # Simple greeting
    if w == ["HELLO"]:
        return "Hello."

    # Goodbye
    if w == ["BYE-BYE"]:
        return "Goodbye."

    # Thank you / welcome / sorry
    if w == ["THANK YOU"]:
        return "Thank you."

    if w == ["WELCOME"]:
        return "You are welcome."
    
    if w == ["YOU"]:
        return "What about you?"

    if w == ["SORRY"]:
        return "I'm Sorry."

    # I am deaf / hearing
    if w == ["I", "DEAF"]:
        return "I am deaf."

    if w == ["I", "MUTE"]:
        return "I am mute."
    
    if w == ["YOU", "DEAF"]:
        return "Are you deaf?"

    if w == ["I", "HEARING"]:
        return "I am hearing."

    # You are hearing (if you ever use this pattern)
    if w == ["YOU", "HEARING"]:
        return "You are hearing."

    # She is a teacher
    if w == ["SHE", "TEACHER"]:
        return "She is a teacher."

    # I am fine (support both patterns you might use)
    if w == ["I'M FINE"] or w == ["I", "I'M FINE"]:
        return "I am fine."

    # Location: I am at home / work
    if w == ["I", "HOME"]:
        return "I am at home."

    if w == ["I", "WORK"]:
        return "I am at work."

    if w == ["YOU","HOME", "WORK"]:
        return "Are you at home or at work?"
    
    # Food: I had food
    if w == ["I", "FOOD"] or w == ["FOOD", "I"]:
        return "I had food."
    
    # Food: I had food
    if w == ["YOU", "FOOD"] or w == ["FOOD", "YOU"]:
        return "You had food?"
    
    # Repeat: sorry, repeat again
    if w == ["SORRY", "AGAIN"]:
        return "Sorry, can you repeat that again?"

    # You can keep adding more patterns below like:
    # if w == ["I", "NEED", "HELP"]:
    #     return "I need help."

    # --------- Default: just clean up the raw words ---------
    return clean_sentence(words)

