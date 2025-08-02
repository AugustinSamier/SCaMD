################ Model loading.

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_model(model_id="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Loads the model.

    Arguments:
    -----------
    - model_id: str,
        Path to the model e.g. "meta-llama/Llama-3.1-8B-Instruct".
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # dispatch on GPU
        torch_dtype=torch.float16
    )
    model.eval()
    return pipeline("text-generation", model=model, tokenizer=tokenizer),tokenizer

pipe,tokenizer=load_model()

################ Functions.

import json, re, gc
import numpy as np

import torch, Stemmer, bm25s
from rank_bm25 import BM25Okapi
from sacrebleu.metrics import BLEU, CHRF
from comet import load_from_checkpoint, download_model
from huggingface_hub import hf_hub_download
from datasets import load_dataset

#CompTra imports
from comptra.utils import _stop_at_stop_token
from comptra.prompts.decompose import get_divide_prompt
from comptra.prompts.merge import get_merge_prompt
from comptra.data.dataset import get_datasets
from comptra.languages import MAPPING_LANG_TO_KEY

def get_datasets(split,dataset_name_or_path: str, language: str, tgt_language: str = None):
    if dataset_name_or_path == "openlanguagedata/flores_plus":
        src_ds = load_dataset(dataset_name_or_path, name=language, split=split)
        tgt_ds = load_dataset(dataset_name_or_path, name=tgt_language, split=split)
        return {
            split: {
                "src": src_ds,
                "tgt": tgt_ds,
            }
        }

STOP_WORDS = [
    "###",
    "\n" * 5,
    "\n\n---",
    "://:",
    "://",
    "____",
    "....",
    ". . . .",
    "strong>strong>",
    "Q:",
    "\nProblem:",
    "://",
    "\nA:",
    "<|eot_id|>",
    "<|start_header_id|>",
    "\n\nFinal Answer:",
    "\n\nProblem:",
    "\n\nInput:",
    "#include",
    "[INST]",
    "\nHuman:",
    "\nNote:",
    "<end_of_turn>",
    "<EOS_TOKEN>",
    "assistant<|end_header_id|>",
]

LLAMA_STOP=[
    "###",
    "\n" * 5,
    "\n\n---",
    "://:",
    "://",
    "____",
    "....",
    ". . . .",
    "strong>strong>",
    "Q:",
    "\nProblem:",
    "://",
    "\nA:",
    "<|eot_id|>",
    "<|start_header_id|>",
    "\n\nFinal Answer:",
    "\n\nProblem:",
    "\n\nInput:",
    "#include",
    "[INST]",
    "\nHuman:",
    "\nNote:",
    "<end_of_turn>",
    "<EOS_TOKEN>",
    "assistant<|end_header_id|>",
    "(Note:",
    "Note:",
    "</Demonstration>",
    "</Demonstrations>",
    "(Option",
    "\n"*2,
    "Traduction:",
    "�",
    "��",
    "���"
    "(Note:"
]

class Retriever:
    """
    Class which defines a Retriever
    """
    def __init__(
        self,
        source_language: str = "English",
        dataset_name_or_path: str = "flores",
        retriever_type: str = "bm25s",
        target_language: str = "French",
        variant: str = "robertson",
        ds_src = None,
        ds_tgt = None,
        seed = 122,
        path = None
    ) -> None:

        self.ds_src = ds_src if ds_src else get_datasets("dev",dataset_name_or_path, source_language)["dev"]["src"]
        self.ds_tgt = ds_tgt if ds_tgt else get_datasets("dev",dataset_name_or_path, target_language)["dev"]["tgt"]
        self.retriever_type = retriever_type

        if "bm25s" in retriever_type:
            corpus = [example["text"] for example in self.ds_src]
            # optional: create a stemmer
            # stemmer = Stemmer.Stemmer("english")
            try:
                stemmer = Stemmer.Stemmer(source_language.lower())
            except Exception as e:
                stemmer = Stemmer.Stemmer("english")
            # tokenize the corpus and only keep the ids (faster and saves memory)
            retriever = bm25s.BM25(
                method="robertson" if variant is None else variant,
                k1=1.5,
                b=0.7,
                delta=1.5 if variant == "bm25+" else None,
            )
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
            # corpus_tokens = bm25s.tokenize(corpus, stopwords=source_language[:2].lower(), stemmer=stemmer)
            retriever.index(corpus_tokens)
            self.retriever = retriever
            self.stemmer = stemmer

        elif "bm25" in retriever_type:
            corpus = [example["text"] for example in self.ds_src]
            bm25 = BM25Okapi([sentence.split(" ") for sentence in corpus])
            self.retriever = bm25

        elif "Random" in retriever_type:
            self.rng = np.random.default_rng(seed)
        else:
            raise ValueError(f"{retriever_type} is not supported!")

    def query(self,sentence: str,k: int,idx_sentence: int = None,level: int = None):
        if "bm25s" in self.retriever_type:
            query_tokens = bm25s.tokenize(sentence, stemmer=self.stemmer)
            results, scores = self.retriever.retrieve(query_tokens, corpus=None, k=k)
            indices = list(results[0])[::-1]  # Least similar to most similar
            indices = [int(element) for element in indices]
            demonstrations = [
                (self.ds_src[i]["text"], self.ds_tgt[i]["text"])
                for i in indices
            ]
            return demonstrations

        elif "bm25" in self.retriever_type:
            scores = self.retriever.get_scores(sentence.split(" "))
            scores = list(scores)
            indices = np.argsort(scores)[-k:]
            demonstrations = [
                (self.ds_src[i]["text"], self.ds_tgt[i]["text"])
                for i in indices
            ]
            return demonstrations

        elif "Random" in self.retriever_type:
            indices = self.rng.choice(len(self.ds_src), size=k, replace=False).tolist()
            demonstrations = [
                (self.ds_src[i]["text"], self.ds_tgt[i]["text"])
                for i in indices
            ]
            return demonstrations
        else:
            pass


########

def decomposition(sentence,divideMethod="vanilla"):
    """
    Decompose the input sentence into smaller subparts.

    Arguments:
    -----------
    - sentence: str,
        The source sentence.
    - divideMethod: str,
        Prompt used for the decomposition e.g. "vanilla", "paraphrase".
    """
    decPrompt=get_divide_prompt(sentence,divideMethod)
    decomposition = pipe(decPrompt, max_new_tokens=1500, do_sample=False,pad_token_id=tokenizer.eos_token_id)
    generated_only = decomposition[0]['generated_text'][len(decPrompt):].lstrip()
    outputDecomp=_stop_at_stop_token(generated_only,STOP_WORDS).strip().split("\n")
    decomp=[]
    for i in range(len(outputDecomp)):
        text=outputDecomp[i].strip().split(".",1)
        decomp.append(text[1].lstrip())
    return decomp


def similarite(decomp,k,retriever):
    """
    Retrieve k example for each sentence in decomp selected by similarity.

    Arguments:
    -----------
    - decomp: list,
        List containing the different subparts of the source sentence.
    - k: int,
        Number of similarity examples to retrieve.
    """
    demonstrationsSRC=[]
    demonstrationsTGT=[]

    for u in range(len(decomp)):
        ret=retriever.query(decomp[u],k)
        demoSRC=[]
        demoTGT=[]
        
        for i in range(k):
            demoSRC.append(ret[i][0])
            demoTGT.append(ret[i][1])
        demonstrationsSRC.append(demoSRC)
        demonstrationsTGT.append(demoTGT)

    demonstrations=[]
    demonstrations.append(demonstrationsSRC)
    demonstrations.append(demonstrationsTGT)
    return demonstrations

def generateCodLlama(sentence,src,langs):
    """
    Generate source-languages pairs in different languages used for the multilingual dictionary by creating a prompt and requesting the model.

    Arguments:
    -----------
    - sentence: str,
        The source sentence.
    - src: str,
        The source language e.g. English.
    - langs: list,
        List of languages used to create the self generated dictionary, e.g. ["French","German","Portuguese"].
    """
    extracts=[]
    for lang in langs:
        extractPrompt = f"""
(1) Please translate the following {src} sentence into {lang}.
(2) Then create a dictionary where each word in the {lang} translation is annotated with its {src} meaning, word-for-word and in order.
(3) The format must be:
{src}: <sentence>
{lang}: <translation>
Dictionary: word1 ({src}1), word2 ({src}2), ...

(4) Example:
English: The cat sleeps on the couch.
French: Le chat dort sur le canapé.
Dictionary: Le (The), chat (cat), dort (sleeps), sur (on), le (the), canapé (couch)

(5) End your response **after the dictionary**. Do not include any explanations or follow-up messages.

Now translate:
{src}: {sentence}
{lang}:
"""
        extract=pipe(extractPrompt, max_new_tokens=500, do_sample=False,pad_token_id=tokenizer.eos_token_id)[0]["generated_text"][len(extractPrompt):]
        extracts.append(extract)
    return extracts

def filtrageTradLLama(translation):
    """
    Localise the real translated sentence in the output by checking every word before it and comparing them to a stop word list. Then check for stop word in the rest of the sentence to return the filtred input.

    Arguments:
    -----------
    - translation: str,
        The output of the model.
    """
    #forward filter
    firstWord=translation.split(" ")[0]
    detectLineWord=firstWord.strip().split("\n")
    if firstWord.strip() in LLAMA_STOP:
        filtred=filtrageTradLLama(translation[len(firstWord):])
    else:
        filtred=translation

    #backward filter
    taille=0
    for words in filtred.split(" "):
        if words!="":
            detectLine=words.strip().split("\n")
            if len(detectLine)!=1:
                if detectLine[0] not in LLAMA_STOP:
                    wor=detectLine[0]
                else:
                    wor=detectLine[1]
                taille+=len(wor)
                return filtred[:taille]
            if words.strip() not in LLAMA_STOP:
                if len(words)>=2:
                    if words[0]!="\n":
                        taille+=len(words)+1
                    else:
                        return filtred[:taille].strip()
                else:
                    taille+=len(words)+1
            else:
                return filtred[:taille]
        else:
            return filtred[:taille].strip()
    return filtred[:taille].strip()


def createCodLlama(dicto,src,name=False,mode=False):
    """
    Take a dictionary and return a prompt made from it.

    Arguments:
    -----------
    - dicto: dict,
        Dictionary containing the source-language pairs made by the model.
    - src: str,
        Source language e.g. English.
    - name: bool,
        If True, will tag "(PROPER_NAME)" for each proper name in the dictionary.
    - mode: bool,
        If True, will add "'" around translated word of each pair.
    """
    prompt=""
    for word in dicto:
        if word[0].isupper() and name:
            proper=" (PROPER_NAME)"
        else:
            proper=""
        if mode:
            prompt+=f"{src}: '{word}'{proper}"
        else:
            prompt+=f"{src}: {word}{proper}"
        for equivalences in dicto[word]:
            if mode:
                prompt+=f", {equivalences[1]}: '{equivalences[0]}'"
            else:
                prompt+=f", {equivalences[1]}: {equivalences[0]}"
        prompt+="\n"
    return prompt

def cutCod(prompt,sentence,size=3,adapting=False,goalSize=40):
    """
    Take the dictionary and a sentence and return only the words present in the sentence with a length superior to a chosen parameter.

    Arguments:
    -----------
    - prompt: str,
        A prompt containing the dictionary.
    - sentence: str,
        The sentence we want to adapt the dictionary for.
    - size: int,
        The minimum size of the words we want to keep in the dictionary.
    - adapting: bool,
        If True, will gradually increase the size to fit the goal size.
    - goalSize: int,
        The goal size of the prompt.
    """
    goal=False
    while goal is False:
        final=[]
        sentence=re.sub(r"[,.\n']"," ",sentence)
        sentence=re.sub(r'["]'," ",sentence)
        lines=prompt.strip().split("\n")
        for line in lines:
            words=line.split(" ")
            first=re.sub(r"[',]","",words[1])
            if len(first)>=size and re.search(rf"\b{re.escape(first)}\b", sentence):
                final.append(line)
        if len(final)<=goalSize or adapting is False:
            goal=True
        else:
            size+=1
    return "\n".join(final)

def newFilterCod(extracts,langs):
    """
    Takes the different generations from the model when creating the multilingual dictionary, extract the different source-languages pairs and return a multilingual dictionary.

    Artguments:
    -----------
    - extracts: list,
        A list containing the generations of the model corresponding to the different HRLs selected.
    - langs: list,
        A list of languages (HRLs) selected to self-generate the multilingual dictionary.
    """
    dicto={}
    for e,extr in enumerate(extracts):
        dictionaries=extr.strip().split("Dictionary:")[1:]
        for d,dictio in enumerate(dictionaries):
            part=dictio.lstrip().split("\n")[0].strip()
            searchPart=part.split(",")

            goodPart=False
            for word in searchPart:
                if any(stop in word for stop in LLAMA_STOP):
                    break
                if "(" in word and ")" in word:
                    eng=word.strip().split("(")[1].split(")")[0]
                    eng=re.sub(r"[,]","",eng).strip()
                    if not eng:
                        continue
                    goodPart=True
                    language=word.strip().split("(")[0].strip()
                    if eng[0].islower():
                        language=language.lower()
                    comb=[language,langs[e]]
                    if eng not in dicto:
                        dicto[eng]=[comb]
                    else:
                        doublon=False
                        for combinations in range(len(dicto[eng])):
                            if comb==dicto[eng][combinations]:
                                doublon=True
                                break
                        if not doublon:
                            dicto[eng].append(comb)
            if goodPart:
                break
    return dicto

def repeatLine(sentence):
    """
    Filter similar lines in the model output.

    Argument:
    -----------
    - sentence: str,
        The output of the model we want to filter.
    """
    lines=sentence.strip().split("\n")
    if len(lines)!=1:
        for line in range(len(lines)):
            if line!=len(lines)-1:
                if lines[line]==lines[line+1]:
                    return "\n".join(lines[:line]).strip()
    return sentence.strip()

def filterRepeat(sentence,filterSize):
    """
    Filter word repetition in the model output.

    Arguments:
    -----------
    - sentence: str,
        The output of the model we want to filter.
    - filterSize: int,
        Determinate max word window repetition filtered.
    """
    sentence=sentence.split("\n")[0].strip()
    length=[]
    words=sentence.split(" ")
    for windSize in range(1,filterSize+1):
        succession=max(2,round(6-windSize))
        for word in range(len(words)-windSize*succession+1):
            repe=True
            actualWind=words[word:word+windSize]
            for success in range(1,succession):
                succWind=words[word+windSize*success:word+(success+1)*windSize]
                if succWind!=actualWind:
                    repe=False
                    break
            if repe:
                length.append(len(" ".join(words[:word+windSize])))
                
    if len(length)!=0:
        return sentence[:min(length)]
    return sentence

def filterChr(sentence,filterSize):
    """
    Filter character repetition in the model output.

    Arguments:
    -----------
    - sentence: str,
        The output of the model we want to filter.
    - filterSize: int,
        Determinate max character window repetition filtered.
    """
    words=sentence.strip().split(" ")
    for w,word in enumerate(words):
        for ch in range(len(word)):
            rep=0
            if ch<len(word)-filterSize:
                for c in range(filterSize):
                    if word[ch]==word[ch+c]:
                        rep+=1
            if rep==5:
                if w!=0:
                    return sentence[:len(" ".join(words[:w]))]
                else:
                    return ""
    return sentence

def insider(prompt,cod):
    """
    Place the multilingual dictionary in-between each demonstration, with the words included in the demonstration.

    Arguments:
    -----------
    - prompt: str,
        Prompt containing the previous demonstrations and the sentence to translate.
    - cod: str,
        Prompt containing the multilingual dictionary.
    """
    if cod:
        codLines=cod.strip().split("\n")
    
        before,demoPart,after=re.split(r"<Demonstrations>|</Demonstrations>",prompt)
        blocPart=demoPart.strip().split("\n\n")
        blocs=[]
        for i,bloc in enumerate(blocPart):
            eng=bloc.split("\n")[1].strip()
            matchingcod=[]
            for w,word in enumerate(codLines):
                codWord=re.sub(r"'","",word.split(",")[0].split("English: ")[1])
                codWord=codWord.replace(" (PROPER_NAME)","")
                if re.search(rf"\b{re.escape(codWord)}\b", eng):
                    matchingcod.append(word)
            if matchingcod:
                bloc1="\n".join(bloc.split("\n")[:2])
                bloc2="\n".join(bloc.split("\n")[2:])
                blocs.append(bloc1+"\n<Dictionary>\n"+"\n".join(matchingcod)+"\n</Dictionary>\n"+bloc2)
            else:
                blocs.append(bloc)
        finalPrompt=before+"\n\n".join(blocs)+after      
    else:
        return prompt
    return finalPrompt
    
def reformPrompt(mergePrompt,codPrompt,place):
    """
    Place the multilingual dictionary in the prompt compared to the demonstrations.

    Arguments:
    -----------
    - mergePrompt: str,
        Prompt containing the previous demonstrations and the sentence to translate.
    - codPrompt: str,
        Prompt containing the multilingual dictionary.
    - place: str,
        Position of the dictionary compared to the demonstrations e.g. "before", "after" or "inside"
    """
    if place=="after":
        parts=mergePrompt.split("</Demonstrations>")
        parts[1]="\n\n<Dictionary>\n"+codPrompt+"\n</Dictionary>\n\n"+parts[1]
        return "</Demonstrations>".join(parts)
    elif place=="before":
        return "<Dictionary>\n"+codPrompt+"\n</Dictionary>\n\n"+mergePrompt
    elif place=="inside":
        return insider(mergePrompt,codPrompt)


def llamaFilter(translation,filterSize=40,filterChrSize=5,filter=True):
    """
    Output filter adapted to LLaMA 3.1 8B Instruct output.

    Arguments:
    -----------
    - translation: str,
        The output of the model we want to filter.
    - filterSize: int,
        Determinate max word window repetition filtered by the function filterRepeat.
    - filterChrSize: int,
        Determinate max character window repetition filtered by the function filterChr.
    - filter: bool,
        If False, will not filter the input.
    """
    if filter is False:
        return translation
    translation=translation.strip()
    newSent=filtrageTradLLama(translation)
    filtred=repeatLine(newSent)
    final=filterRepeat(filtred,filterSize)
    fin=filterChr(final,filterChrSize)
    if len(fin)!=0:
        if fin[-1]==".":
            fin=fin[:-1]
    fin=re.sub(r"�","",fin)
    return fin.strip()


def SCaMD(
        sentence,
        langs,
        k,
        src,
        tgt,
        retriever,
        scamd=True,
        comptra=True,
        prompts=None,
        outPrompts=False,
        cod=None,
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
    SCaMD pipeline decomposes the input sentence into smaller parts, retrieve similar examples in the dataset dev set to use them as few-shot example to translate each subparts of the source sentence. 
    A multilingual dictionary is self-generated from the source sentence into HRLs.
    The translated parts and the multilingual dictionary are then merged in few-shot example prompt with lexical augmentation to finally translate the input sentence.

    Arguments:
    -----------
    - sentence: str,
        Input sentence.
    - langs: list,
        List of languages used to create the self generated dictionary, e.g. ["French","German","Portuguese"].
    - k: int,
        Number of similarity example retrieved.
    - src: str,
        Source language e.g. English.
    - tgt: str,
        Target language e.g. Amharic.
    - retriever: Retriever,
        Retriever used to find similar examples.
    - scamd: bool,
        If False, will skip the multilingual dictionaries part in the pipeline. Only set on False if you already have generated multilingual dictionaries.
    - comptra: bool,
        If False, will skip the compositional-translation part in the pipeline. Only set on False if you already have generated compositional-translation prompts.
    - prompts: List,
        List containing the compositional-translation prompts generated in the past. Argument used to save time.
    - outPrompts: bool,
        If True, will return a file containing the compositional-translation prompts generated.
    - cod: List,
        List containing the multilingual dictionaries generated in the past. Argument used to save time.
    - outCod: bool,
        If True, will return a file containing the multilingual dictionaries generated.
    - filterSize: int,
        Determinate max word window repetition filtered by the function filterRepeat.
    - filterChrSize: int,
        Determinate max character window repetition filtered by the function filterChr.
    - name: bool,
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
    
    if prompts is None and comptra:
        decomp=decomposition(sentence)
        torch.cuda.empty_cache()
        gc.collect()
        demonstrations=similarite(decomp,k,retriever)
        
        decompTrads=[]
        torch.cuda.empty_cache()
        gc.collect()
        
        for i,morceaux in enumerate(decomp):
            mergePrompt=get_merge_prompt(morceaux,demonstrations[0][i],demonstrations[1][i],src,tgt)
            traductionMorceau=pipe(mergePrompt, max_new_tokens=500, do_sample=False,pad_token_id=tokenizer.eos_token_id)[0]["generated_text"][len(mergePrompt):].strip()
            decompTrads.append(llamaFilter(traductionMorceau,filterSize=filterSize,filterChrSize=filterChrSize))
            
        torch.cuda.empty_cache()
        gc.collect()
        mergeFinalPrompt=get_merge_prompt(sentence,decomp,decompTrads,src,tgt)
    else:
        mergeFinalPrompt=prompts
    if outPrompts:
        return "|SCaMD|".join(mergeFinalPrompt)
    if scamd:
        if cod:
            extractsFin=cod
        else:
            extractsFin=generateCodLlama(sentence,src,langs)
        if outCod:
            return extractsFin
        dictoFin=newFilterCod(extractsFin,langs)
        promptCodFin=createCodLlama(dictoFin,src,name=nameMode,mode=mode)
        mergeFinalPrompt=reformPrompt(mergeFinalPrompt,cutCod(prompt=promptCodFin,sentence=sentence,size=size,adapting=adapting,goalSize=goalSize),place=place)

    traductionFinale=pipe(mergeFinalPrompt, max_new_tokens=500, do_sample=False,pad_token_id=tokenizer.eos_token_id)[0]["generated_text"][len(mergeFinalPrompt):].strip()
    
    traductionFinale=llamaFilter(traductionFinale,filterSize=filterSize,filterChrSize=filterChrSize,filter=filter)
    return traductionFinale