import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import argparse
import torch
import gc
from scamd_utils import SCaMD,Retriever,get_datasets

def sampler(
    src="eng_Latn",
    tgt="amh_Ethi",
    k=5,
    codLangs=["French","German","Portuguese"],
    ds="flores",
    checkpoint=0,
    path_output="Generation1",
    nb=None,
    comptra=True,
    scamd=True,
    prompts_path=None,
    outPrompts=False,
    dictionaries_path=None,
    outCod=False,
    filterSize=40,
    filterChrSize=5,
    nameMode=False,
    mode=True,
    size=1,
    adapting=True,
    goalSize=40,
    place="before",
    filter=True
):
    
    """
    Function executing the SCaMD pipeline.

    Arguments:
    -----------
        - src: str,
            Source language e.g. English.
        - tgt: str,
            Target language e.g. Amharic.
        - k: int,
            Number of similarity example retrieved.
        - codLangs: list,
            list of languages used to create the self generated dictionary, e.g. ["French","German","Portuguese"].
        - ds: str,
            Dataset used e.g. flores.
        - checkpoint: int,
            Number of the sentence you want to start translating from.
        - path_output: str,
            Name or path to the folder containing the generations (if not existing it will be created).
        - nb: int,
            Number of sentences you want to translate from your checkpoint. If None, will do the whole dataset.
        - comptra: bool,
            If False, will skip the compositional-translation part in the pipeline. Only set on False if you already have generated compositional-translation prompts.
        - scamd: bool,
            If False, will skip the multilingual dictionaries part in the pipeline. Only set on False if you already have generated multilingual dictionaries.
        - prompts_path: Path,
            Path to the file containing the compositional-translation prompts generated in the past. Argument used to save time.
        - outPrompts: bool,
            If True, will return a file containing the compositional-translation prompts generated. Will only return the prompts not the final translations nor the multilingual dictionaries.
        - dictionaries_path: Path,
            Path to the file containing the multilingual dictionaries generated in the past. Argument used to save time.
        - outCod: bool,
            If True, will return a file containing the multilingual dictionaries generated. Will only return the multilingual dictionaries not the final translations nor the prompts.
        - filterSize: int,
            Determinate max word window repetition filtered by the function filterRepeat.
        - filterChrSize: int,
            Determinate max character window repetition filtered by the function filterChr.
        - nameMode: bool,
            If True, will tag "(PROPER_NAME)" for each proper name in the dictionary.
        - mode: bool,
            If True, will add "'" around translated word of translation-pair in the dictionary.
        - size: int,
            The minimum size of the words we want to keep in the dictionary.
        - adapting: bool,
            If True, will gradually increase the size to fit the dictionary goal size.
        - goalSize: int,
            The goal size of the dictionary prompt.
        - place: str,
            Position of the dictionary compared to the demonstrations e.g. "before", "after" or "inside".
        - filter: bool,
            If True, will filter the final output of the pipeline (recommanded).
    """
    

    ds_dict = get_datasets("devtest","openlanguagedata/flores_plus", src, tgt)
    ds_src = ds_dict["devtest"]["src"]
    ds_tgt = ds_dict["devtest"]["tgt"]
    retriever = Retriever(
            retriever_type="bm25s",
            source_language=src,
            target_language=tgt,
            dataset_name_or_path="openlanguagedata/flores_plus"
        )
    if nb is None:
        nb=len(ds_src["text"])
    
    ### Loading the existing prompts and dictionaries in order to save them.
    path_name="out/GENERATIONS/"+path_output
    os.makedirs(path_name, exist_ok=True)
    
    if outCod:
        output_path = os.path.join(path_name, f"{ds}-Dictionaries-({src}-{codLangs[0]}-{codLangs[1]}-{codLangs[2]}).jsonl")
    elif outPrompts:
        output_path = os.path.join(path_name, f"{ds}-Compositional-translation-{src}.jsonl")
    else:
        output_path = os.path.join(path_name, f"{ds}-{src}-{tgt}-generations.jsonl")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = [json.loads(line) for line in f]
        print(f"{len(loaded)} elements reloaded from {output_path}")
        checkpoint=len(loaded)
    else:
        print("No existing file found, creating a new one.")

    ### Loading part for prompts and dictionaries if generated in the past (used to save time).
    if prompts_path:
        prompts_path=os.path.join("Prompts/Compositional-translation/",prompts_path)
        promptComp = []
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                prompt = json.loads(line)
                promptComp.append(prompt)
        
        promptComp=promptComp[checkpoint:checkpoint+nb]
    
    if dictionaries_path:
        dictionaries_path=os.path.join("Prompts/Dictionaries/",dictionaries_path)
        cods=[]
        with open(dictionaries_path, "r", encoding="utf-8") as f:
            for line in f:
                cod = json.loads(line)
                cods.append(cod.split("|SCaMD|"))
        
        cods=cods[checkpoint:checkpoint+nb]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[INFO] Using device: {device}")
    for i in range(checkpoint,nb):
        torch.cuda.empty_cache()
        gc.collect()
        promptIndex=promptComp[i] if prompts_path else None
        codsIndex=cods[i] if dictionaries_path else None
            
        output=SCaMD(
            sentence=ds_src[i]["text"],
            langs=codLangs,
            k=k,
            src=src,
            tgt=tgt,
            scamd=scamd,
            comptra=comptra,
            prompts=promptIndex,
            outPrompts=outPrompts,
            cod=codsIndex,
            outCod=outCod,
            filterSize=filterSize,
            filterChrSize=filterChrSize,
            nameMode=nameMode,
            mode=mode,
            size=size,
            adapting=adapting,
            goalSize=goalSize,
            place=place,
            retriever=retriever,
            filter=filter
        )
        
        if outPrompts or outCod:
            if outPrompts:
                print(f"Prompt {i+1}/{nb}")
            if outCod:
                print(f"Dictionary {i+1}/{nb}")
                
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
        else:
            print(f"Sentence number : {i+1}\nSentence : {ds_src[i]["text"]}\n")
            print(f"Translation generated: {output}")
            print("\n\n")
            with open(output_path, "a", encoding="utf-8") as f:
                line = {
                    "source": ds_src[i]["text"],
                    "hypothesis": output,
                    "reference": ds_tgt[i]["text"]
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("Process finished.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--src",type=str,default="eng_Latn")
    parser.add_argument("--tgt",type=str,default="amh_Ethi")
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--CodLangs",type=list,default=["French","German","Portuguese"])
    parser.add_argument("--ds",type=str,default="flores")
    parser.add_argument("--checkpoint",type=int,default=0)
    parser.add_argument("--path_output",type=str,default="Generations1")
    parser.add_argument("--nb",type=int,default=None)
    parser.add_argument("--comptra",type=bool,default=True)
    parser.add_argument("--scamd",type=bool,default=True)
    parser.add_argument("--prompts_path",type=str,default=None)
    parser.add_argument("--outPrompts",type=bool,default=False)
    parser.add_argument("--dictionaries_path",type=str,default=None)
    parser.add_argument("--outCod",type=bool,default=False)
    parser.add_argument("--filterSize",type=int,default=40)
    parser.add_argument("--filterChrSize",type=int,default=5)
    parser.add_argument("--nameMode",type=bool,default=False)
    parser.add_argument("--mode",type=bool,default=True)
    parser.add_argument("--size",type=int,default=1)
    parser.add_argument("--adapting",type=bool,default=True)
    parser.add_argument("--goalSize",type=int,default=40)
    parser.add_argument("--place",type=str,default="before")
    parser.add_argument("--filter",type=bool,default=True)
    args=parser.parse_args()
    sampler(
        src=args.src,
        tgt=args.tgt,
        k=args.k,
        codLangs=args.CodLangs,
        ds=args.ds,
        checkpoint=args.checkpoint,
        path_output=args.path_output,
        nb=args.nb,
        comptra=args.comptra,
        scamd=args.scamd,
        prompts_path=args.prompts_path,
        outPrompts=args.outPrompts,
        dictionaries_path=args.dictionaries_path,
        outCod=args.outCod,
        filterSize=args.filterSize,
        filterChrSize=args.filterChrSize,
        nameMode=args.nameMode,
        mode=args.mode,
        size=args.size,
        adapting=args.adapting,
        goalSize=args.goalSize,
        place=args.place,
        filter=args.filter
    )