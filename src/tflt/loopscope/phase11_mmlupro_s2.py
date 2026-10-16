"""Opt-in, pre-outcome MMLU-Pro adaptation of the S2 final-line parser."""
import re

PARSER_VERSION = "S2-MMLU-Pro-v1-20261016"
# S1/S2 line grammar, with only the recognized alphabet expanded from A-D to A-Z.
PATTERN = r"final[ \t]+answer[ \t]*[:：][ \t]*(?:\([ \t]*([A-Z])[ \t]*\)|([A-Z]))[ \t]*[.!;,．！。]?"


def extract_mmlupro_s2(text, option_count):
    """Parse generated text alone; any syntactically matched illegal label fails.

    Legal labels are the first ``option_count`` Latin capitals. An illegal
    matched label takes precedence over legal-label conflict. Repeated legal
    labels are accepted only when all normalize to the same letter.
    """
    if not isinstance(text, str) or type(option_count) is not int or not 1 <= option_count <= 26:
        raise ValueError("S2-MMLU-Pro requires response text and an actual option count in 1..26")
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count]
    letters = []
    for line in text.splitlines():
        match = re.fullmatch(PATTERN, line.strip(), re.IGNORECASE | re.ASCII)
        if match:
            letters.append((match[1] or match[2]).upper())
    illegal = [letter for letter in letters if letter not in labels]
    distinct = set(letters)
    status = ("no_match" if not letters else "out_of_range" if illegal else
              "conflict" if len(distinct) != 1 else "ok")
    predicted = letters[0] if status == "ok" else None
    return {"parser_version": PARSER_VERSION, "option_count": option_count,
            "extraction_status": status,
            "answer_mode": ("single" if len(letters) == 1 else "consistent_multiple"
                            if status == "ok" else status),
            "matched_final_line_count": len(letters), "matched_letters": letters,
            "out_of_range_line_count": len(illegal),
            "predicted_label": predicted,
            "prediction_index": labels.index(predicted) if predicted is not None else None}
