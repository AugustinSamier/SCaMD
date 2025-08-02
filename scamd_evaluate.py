import json
import argparse
import evaluate
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU,CHRF
from scamd_utils import llamaFilter
import gc

def load_data(jsonl_path,activeFilter):
    sources,preds,refs=[],[],[]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            sources.append(ex["source"])
            preds.append(llamaFilter(ex["hypothesis"],filter=activeFilter))
            refs.append(ex["reference"])
    return sources, preds, refs

def evaluate_bleu(preds, refs):
    b=bleu.corpus_score(preds,[refs]).score
    return b

def evaluate_chrf(preds, refs):
    c=chrf.corpus_score(preds,[refs]).score
    return c


def evaluate_comet(sources, preds, refs):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, preds, refs)]
    score= model.predict(data, batch_size=8, gpus=0)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to jsonl generation file")
    parser.add_argument("--activeFilter", type=bool,default=True)
    args = parser.parse_args()
    ds=args.file.split("/")[-1].split("-")[0]

    sources, preds, refs = load_data(
        jsonl_path=args.file,
        activeFilter=args.activeFilter
        )

    if ds=="flores":
        bleu=BLEU(tokenize="flores200")
    chrf=CHRF(word_order=2)
    bleu_score = evaluate_bleu(preds,refs)
    chrf_score = evaluate_chrf(preds,refs)
    comet_score=evaluate_comet(sources,preds,refs)

    print(f"BLEU: {bleu_score:.4f}")
    print(f"chrF: {chrf_score:.4f}")
    print(f"COMET-22: {float(comet_score["system_score"]):.4f}")
