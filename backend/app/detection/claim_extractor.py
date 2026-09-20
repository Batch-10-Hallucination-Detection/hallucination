import spacy

nlp = spacy.load("en_core_web_sm")


def split_clauses(sentence):
    """
    Split a sentence into simple clauses using coordinating conjunctions
    such as 'and', 'but', and 'or'.
    """

    clauses = []

    for token in sentence:
        if token.dep_ == "cc" and token.text.lower() in ["and", "but", "or"]:
            left = sentence.text[:token.idx - sentence.start_char].strip()

            right_start = token.idx + len(token.text) - sentence.start_char
            right = sentence.text[right_start:].strip()

            if left and right:
                clauses.append(left)
                clauses.append(right)

                return clauses

    return [sentence.text.strip()]


def make_self_contained(clause, subject):
    """
    Replace simple pronoun-based subjects with the main subject.
    """

    if not clause:
        return clause

    words = clause.split()

    if not words:
        return clause

    first_word = words[0].lower()

    pronouns = [
        "it",
        "they",
        "he",
        "she",
        "this",
        "that"
    ]

    if first_word in pronouns and subject:
        return subject + " " + " ".join(words[1:])

    return clause


def extract_claims(text):
    """
    Extract atomic and self-contained claims from text.

    Steps:
    1. Split text into sentences.
    2. Identify the main subject.
    3. Split simple compound sentences into clauses.
    4. Make simple pronoun-based clauses self-contained.
    """

    doc = nlp(text)

    claims = []

    for sentence in doc.sents:

        sentence_text = sentence.text.strip()

        if not sentence_text:
            continue

        # Find the main subject
        main_subject = None

        for token in sentence:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                main_subject = token.text
                break

        # Split sentence into clauses
        clauses = split_clauses(sentence)

        # Convert clauses into self-contained claims
        for clause in clauses:

            clause = clause.strip()

            if not clause:
                continue

            clause = make_self_contained(
                clause,
                main_subject
            )

            claims.append(clause)

    return claims


if __name__ == "__main__":

    text = """
    Metformin is used to treat type 2 diabetes and it reduces blood glucose levels.
    Python was created by Guido van Rossum and it was first released in 1991.
    """

    print("\nCLAUSE PARSER + ATOMIC CLAIM EXTRACTION\n")

    claims = extract_claims(text)

    for i, claim in enumerate(claims, 1):
        print(f"Claim {i}: {claim}")