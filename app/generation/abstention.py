"""'Not found in the sources' responses."""


def abstain_message(question: str, queries_tried: list[str]) -> str:
    msg = "I couldn't find enough information in the available sources to answer this reliably."
    if queries_tried:
        tried = "; ".join(f'"{q}"' for q in queries_tried[:5])
        msg += f" I searched for: {tried}."
    return msg + " You could rephrase the question or add the relevant documents to the corpus."