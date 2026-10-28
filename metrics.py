from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["pos", "neu", "neg", "q"]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_f1_3": f1_score(y_true, y_pred, labels=["pos", "neu", "neg"], average="macro", zero_division=0),
        "valid_rate": sum(p in LABELS for p in y_pred) / len(y_pred),
    }


def print_report(y_true: list[str], y_pred: list[str]) -> None:
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=3))
    print("confusion matrix (แถว = คำตอบจริง, คอลัมน์ = ที่โมเดลทาย)")
    print("      " + "".join(f"{label:>6}" for label in LABELS))
    for label, row in zip(LABELS, confusion_matrix(y_true, y_pred, labels=LABELS)):
        print(f"{label:>5} " + "".join(f"{value:>6}" for value in row))

