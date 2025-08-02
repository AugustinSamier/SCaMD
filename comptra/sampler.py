from typing import List, Tuple
import numpy as np
import torch
import re
import os

from comptra.prompts.maps import get_maps_aspects
from comptra.prompts.step_by_step import get_step_by_step_prompts
from comptra.prompts.tear import get_tear_prompts
from comptra.prompts.merge import get_merge_prompt, extract_translation
from comptra.prompts.refine import get_refine_prompt
from comptra.prompts.decompose import get_divide_prompt
from comptra.prompts.translate import *
from comptra.prompts.templates import Template

from comptra.models import IFT_MODELS
from comptra.utils import (
    hf_generate,
    _stop_at_stop_token,
    remove_repeating_bigram,
    get_best_sentence,
    quality_estimation,
    is_lang,
)
from comptra.languages import MAPPING_LANG_TO_KEY, NON_FLORES
from comptra.apply_chat_template import apply_chat_template

from accelerate import Accelerator

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

MAX_NUMBER_OF_PROPOSITIONS = 8

from comptra.prompts.templates import get_linguistic_prompt


class Sampler:
    """
    Sampler for Compositional Translation (CoTra)
    Arguments
    ---------
        - model_name_or_path: str,
            Name or path to the model of interest e.g. google/gemma-2-2b-it on HF, gpt-3.5-turbo-0125 on OpenAI etc.
        - tokenizer_name_or_path: str,
            Name or path to the tokenizer of interest if relevant. Usually the same as model_name_or_path on HF.
        - src: str,
            Source language e.g. English
        - tgt: str,
            Target language e.g. French
        - template: Template,
            Template used for few-shot MT. Can be set to None.
        - merge_prompt: str,
            Basically the prompting scheme adopted for the merge e.g. vanilla
        - method_translate: str,
            Which method to adopt for the translation e.g. "vanilla", "cot", "maps", "refine" etc.
        - selection_method: str,
        - method_divide: str,
            Prompt used for the decomposition e.g. "vanilla", "paraphrase" etc.

    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        src: str,
        tgt: str,
        template: Template,
        merge_prompt: str,
        method_translate: str,
        selection_method: str,
        nllb_name_or_path: str,
        method_divide: str = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.src = src
        self.tgt = tgt
        self.template = template
        self.merge_prompt = merge_prompt
        self.method_translate = method_translate
        self.selection_method = selection_method
        self.method_divide = method_divide

        self.ift = self.model_name_or_path in IFT_MODELS

        print(
            f"We use {'a pretrained' * (not self.ift) + 'an instruction fine-tuned' * self.ift} model!"
        )
        if method_translate in ["nllb", "cod"]:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.nllb = AutoModelForSeq2SeqLM.from_pretrained(
                nllb_name_or_path,
                load_in_4bit=True if torch.cuda.is_available() else False,
                device_map="auto",
                # device_map=(
                #    {"": Accelerator().process_index}
                #    if torch.cuda.is_available()
                #    else None
                # ),
                # torch_dtype=torch.float16
            )
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name_or_path)
            self.nllb_tokenizer.src_lang = MAPPING_LANG_TO_KEY[self.src]

        if method_translate == "nllb":
            print(f"We are going to use NLLB, precisely {nllb_name_or_path}")
        elif method_translate == "vanilla":
            print(f"We are going to use vanilla zero/few-shot MT.")
        elif method_translate == "cot":
            print(f"We are going to use zero/few-shot CoT MT.")
        elif method_translate == "maps":
            print(f"We are going to use MAPS MT.")
        elif method_translate == "step_by_step":
            print(f"We are going to use Step-by-Step Translation.")
        elif method_translate == "TEaR":
            print("We are going to use TEaR.")
        elif method_translate == "cod":
            print("We are going to use Chain-of-Dictionary (CoD).")
        else:
            pass

        print(
            f"Merge algorithm: {self.merge_prompt}, selection method: {self.selection_method}"
        )

    def update_src(self, src):
        self.src = src

    def update_tgt(self, tgt):
        self.tgt = tgt

    def update_template(self, template):
        self.template = template

    def apply_chat_template(self, prompt):
        """Apply the chat model, useful with IFT models."""
        return apply_chat_template(self.model_name_or_path)(prompt)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        num_return_sequences: int,
        num_beams: int,
        do_sample: bool,
        request_batch_size: int,
        verbose: bool = True,
    ) -> List[List[str]]:
        raise NotImplementedError("The function `generate` is not implemented.")

    def from_source_to_target(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 5,
        do_sample: bool = False,
        verbose: bool = True,
    ) -> List[str]:
        inputs_tensor = self.nllb_tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(self.nllb.device)

        translated_tokens = self.nllb.generate(
            **inputs_tensor,
            # forced_bos_token_id=self.nllb_tokenizer.lang_to_code_id[
            forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(
                MAPPING_LANG_TO_KEY[self.tgt]
            ),
            # ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )

        outputs = self.nllb_tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        if verbose:
            print("===")
            for i, output in enumerate(outputs):
                print(f"{i+1} -> {output}")
            print("===")

        return outputs

    def translate(
        self,
        sentences: List[str],
        demonstrations: List[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ):
        if self.method_translate == "nllb":
            outputs = self.from_source_to_target(
                prompts=sentences,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                verbose=verbose,
            )
            outputs = [remove_repeating_bigram(output) for output in outputs]
            outputs = [
                get_best_sentence(
                    target_translations=outputs[i : i + num_return_sequences],
                    src=self.src,
                    tgt=self.tgt,
                    source_sentence=sentences[i // num_return_sequences],
                    strategy=self.selection_method,
                )
                for i in range(0, len(outputs), num_return_sequences)
            ]
            return outputs
        elif self.method_translate == "vanilla":
            prompts = [
                get_translate_prompt(
                    sentence,
                    src=self.src,
                    tgt=self.tgt,
                    demonstrations=demonstrations[i]
                    if (demonstrations and demonstrations[i])
                    else [],
                    template=self.template,
                    ift=self.ift,
                )
                for i, sentence in enumerate(sentences)
            ]
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]
            print(f"===\n{prompts[0]}\n===")
            outputs = self.generate(
                prompts=prompts,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            final_outputs = []
            for i, output in enumerate(outputs):
                final_outputs.append(
                    get_best_sentence(
                        target_translations=[
                            remove_repeating_bigram(
                                _stop_at_stop_token(output[j], STOP_WORDS)
                                .strip()
                                .split("\n")[0]
                            )
                            for j in range(len(output))
                        ],
                        src=self.src,
                        tgt=self.tgt,
                        source_sentence=sentences[i],
                        strategy=self.selection_method,
                    )
                )
            return final_outputs
        elif self.method_translate == "cot":
            prompts = [
                get_cot_prompt(
                    sentence,
                    src=self.src,
                    tgt=self.tgt,
                    demonstrations=demonstrations[i],
                    template=self.template,
                )
                for i, sentence in enumerate(sentences)
            ]
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]
            print(f"===\n{prompts[0]}\n===")
            outputs = self.cot(
                prompts=prompts,
                sentences=sentences,
                max_new_tokens=max(1500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            return outputs
        elif self.method_translate == "maps":
            outputs = self.maps(
                sentences=sentences,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            return outputs
        elif self.method_translate == "step_by_step":
            outputs = self.step_by_step(
                sentences=sentences,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            return outputs
        elif self.method_translate == "TEaR":
            outputs = self.tear(
                sentences=sentences,
                demonstrations=demonstrations,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            return outputs
        elif self.method_translate == "cod":
            outputs = self.cod(
                sentences=sentences,
                max_new_tokens=max(1500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            return outputs
        elif self.method_translate in ["pos", "morph", "dep", "ner", "none"]:
            prompts = [
                get_linguistic_prompt(
                    sentence,
                    src=self.src,
                    tgt=self.tgt,
                    demonstrations=demonstrations[i],
                    feature=self.method_translate,
                    ift=self.ift,
                )
                for i, sentence in enumerate(sentences)
            ]
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]
            print(f"===\n{prompts[0]}\n===")
            outputs = self.generate(
                prompts=prompts,
                max_new_tokens=min(500, max_new_tokens),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            final_outputs = []
            for i, output in enumerate(outputs):
                final_outputs.append(
                    get_best_sentence(
                        target_translations=[
                            remove_repeating_bigram(
                                _stop_at_stop_token(output[j], STOP_WORDS)
                                .strip()
                                .split("\n")[0]
                            )
                            for j in range(len(output))
                        ],
                        src=self.src,
                        tgt=self.tgt,
                        source_sentence=sentences[i],
                        strategy=self.selection_method,
                    )
                )
            return final_outputs
        else:
            pass

    def divide(
        self,
        n_splits: int,
        sentences: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        """
        Arguments
        ---------
        - sentences: List[str],
            list of sentences to divide into simpler entities.
        - n_splits: int,
            Number of entities to divide into.
        """
        prompts = [
            get_divide_prompt(sentence, self.method_divide) for sentence in sentences
        ]
        # prompts = [self.apply_chat_template(prompt) for prompt in prompts] # Not needed
        outputs = []
        for i in range(0, len(prompts), request_batch_size):
            output = self.generate(
                prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                verbose=verbose,
            )
            # Take the first non-empty element of the list of outputs for each prompt
            for element in output:
                for candidate in element:
                    if candidate.strip() != "":
                        break
                outputs.append(candidate)
        list_of_propositions = []
        for i, output in enumerate(outputs):
            if output.lstrip().startswith("###"):
                output = output.lstrip()[3:]
            if n_splits >= 0:
                output = _stop_at_stop_token(
                    output,
                    STOP_WORDS + [f"\t{n_splits + 1}. ", f"    {n_splits + 1}. ",],
                )
            else:
                output = _stop_at_stop_token(
                    output,
                    STOP_WORDS
                    + [
                        f"\t{MAX_NUMBER_OF_PROPOSITIONS}. ",
                        f"    {MAX_NUMBER_OF_PROPOSITIONS}. ",
                    ],
                )
            propositions = []
            if "    1." in output:
                pattern = r"    (\d|-)\. "
            else:
                pattern = r"(\d)\. "
            splitted_output = re.split(pattern, output)
            for j, proposition in enumerate(splitted_output):
                if j == 0:
                    # everything before the first match
                    continue
                if j % 2 == 1:
                    # matches the iterator
                    continue
                propositions.append(proposition.strip())
            propositions = [
                proposition.strip().split("\n")[0] for proposition in propositions
            ]
            propositions = [
                proposition
                for proposition in propositions
                if len(proposition.strip()) != 0
            ]
            # Remove duplicates
            propositions = list(set(propositions))
            print(propositions)
            if len(propositions) == 0:
                # propositions.append(sentences[i])
                pass
            if n_splits != -1:
                list_of_propositions.append(propositions[0:n_splits])
            else:
                list_of_propositions.append(propositions)

        return list_of_propositions

    def refine(
        self,
        sentences: List[str],
        prev_translations: List[str],
        number_of_refining_steps: int = 1,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ):
        for _ in range(number_of_refining_steps):
            prompts = [
                get_refine_prompt(sentence, prev_translation, self.src, self.tgt)
                for sentence, prev_translation in zip(sentences, prev_translations)
            ]
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]
            print(f"===\n{prompts[0]}\n===")
            outputs = []
            for i in range(0, len(prompts), request_batch_size):
                output = self.generate(
                    prompts[i : i + request_batch_size],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    request_batch_size=request_batch_size,
                    verbose=verbose,
                )
                for j in range(len(output)):
                    outputs.append(output[j])

            assert len(sentences) == len(outputs), "size mismatch in `refine`"
            next_translations = [
                get_best_sentence(
                    target_translations=[
                        remove_repeating_bigram(
                            _stop_at_stop_token(output[j], STOP_WORDS)
                            .strip()
                            .split("\n")[0]
                            .strip()
                        )
                        # output[j].strip().split("\n")[0].strip()
                        for j in range(len(output))
                    ],
                    src=self.src,
                    tgt=self.tgt,
                    source_sentence=sentence,
                    strategy=self.selection_method,
                )
                for sentence, output in zip(sentences, outputs)
            ]
            # Next refinement step
            prev_translations = next_translations

        return prev_translations

    def merge(
        self,
        sentences: List[str],
        inputs: List[List[str]],
        outputs: List[List[str]],
        demonstrations: List[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ):
        prompts = []
        for i, sentence in enumerate(sentences):
            c_in, c_out = [], []
            for j, target in enumerate(outputs[i]):
                if (
                    target.strip() == ""
                    or (
                        not is_lang(target, self.tgt)
                        and self.tgt in MAPPING_LANG_TO_KEY
                        and self.tgt not in NON_FLORES
                    )
                    or (is_lang(target, self.src))
                ):
                    continue
                c_in.append(inputs[i][j])
                c_out.append(target)
            if demonstrations:
                for _, pair in enumerate(demonstrations[i]):
                    a, b = pair
                    c_in.append(a)
                    c_out.append(b)

            prompt = get_merge_prompt(
                sentence=sentence,
                inputs=c_in,
                outputs=c_out,
                src=self.src,
                tgt=self.tgt,
                template=self.template,
                method=self.merge_prompt,
            )
            prompt = self.apply_chat_template(prompt)
            if i == 0:
                print(f"===\n{prompt}\n===")

            prompts.append(prompt)

        outputs = self.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            do_sample=do_sample,
            request_batch_size=request_batch_size,
            verbose=verbose,
        )
        # Postprocess the output
        translations = [
            get_best_sentence(
                target_translations=[
                    remove_repeating_bigram(
                        extract_translation(
                            sentence=_stop_at_stop_token(output[i], STOP_WORDS),
                            method=self.merge_prompt,
                        )
                    )
                    for i in range(len(output))
                ],
                source_sentence=sentence,
                src=self.src,
                tgt=self.tgt,
                strategy=self.merge_prompt,
            )
            for sentence, output in zip(sentences, outputs)
        ]

        return translations

    def cod(
        self,
        sentences: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ):
        translation_prompts = []
        for i in range(0, len(sentences), request_batch_size):
            inputs = sentences[i : min(i + request_batch_size, len(sentences))]
            prompts = [
                self.apply_chat_template(
                    f"Extract the words from the following text: {sentence}\n"
                    # + "Do not include very common or stop words (articles, pronouns, prepositions, conjunctions etc.) and make sure to only return the words as a numbered list (1., 2. etc.), nothing more."
                    + "Make sure to only return the words as a numbered list (1., 2. etc.), nothing more."
                )
                for sentence in inputs
            ]
            outputs = self.generate(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            pattern = r"(?m)^\s*\d+\.\s*(.+)"
            list_of_words = [
                re.findall(pattern, list_of_texts[0].strip())
                for list_of_texts in outputs
            ]
            list_of_words = [
                [word.strip() for word in words if word.strip() != ""]
                for words in list_of_words
            ]
            for words in list_of_words:
                print(words)
            flattened_list_of_words = [
                element for words in list_of_words for element in words
            ]
            print(f"Len(flattened_list_of_words): {len(flattened_list_of_words)}")
            dico_of_translations = {}
            # Translate each word into the three intermediate languages as well as into the target language
            for language in ["French", "German", "Portuguese", self.tgt]:
                outputs = []
                # Batching to avoid Cuda OOM
                for j in range(0, len(flattened_list_of_words), request_batch_size):
                    batch_inputs = flattened_list_of_words[
                        j : min(j + request_batch_size, len(flattened_list_of_words))
                    ]
                    inputs_tensor = self.nllb_tokenizer(
                        batch_inputs, return_tensors="pt", padding=True
                    ).to(self.nllb.device)
                    # print(f"input_tensor.input_ids: {inputs_tensor['input_ids'].shape}")
                    translated_tokens = self.nllb.generate(
                        **inputs_tensor,
                        forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(
                            MAPPING_LANG_TO_KEY[language]
                        ),
                        max_new_tokens=min(30, max_new_tokens),
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        num_return_sequences=num_return_sequences,
                        do_sample=do_sample,
                    )

                    batch_outputs = self.nllb_tokenizer.batch_decode(
                        translated_tokens, skip_special_tokens=True
                    )
                    outputs.extend(batch_outputs)
                dico_of_translations[language] = outputs
            
            prompts = []
            start = 0
            for j in range(len(inputs)):
                prompt = ""
                for k in range(len(list_of_words[j])):
                    src = list_of_words[j][k]
                    fra, deu, pt, tgt = (
                        dico_of_translations["French"][start + k],
                        dico_of_translations["German"][start + k],
                        dico_of_translations["Portuguese"][start + k],
                        dico_of_translations[self.tgt][start + k],
                    )
                    prompt += f'"{src}" means "{fra}" means "{deu}" means "{pt}" means "{tgt}".\n'
                prompt = f"{prompt}\nTranslate the following text from {self.src} into {self.tgt}: {inputs[j]}"
                prompt += "\n\nPlease provide only the translation, nothing more."
                prompt = self.apply_chat_template(prompt)
                start += len(list_of_words[j])
                prompts.append(prompt)
            print(f"{'-' * 50}\n{prompts[0]}\n{'-' * 50}")
            translation_prompts.extend(prompts)
        # Final translation with CoD prompts
        outputs = self.generate(
            prompts=translation_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            do_sample=do_sample,
            request_batch_size=request_batch_size,
            verbose=verbose,
        )
        # Postprocess the output
        translations = [
            get_best_sentence(
                target_translations=[
                    remove_repeating_bigram(
                        extract_translation(
                            sentence=_stop_at_stop_token(output[i], STOP_WORDS),
                            method=self.merge_prompt,
                        )
                    )
                    for i in range(len(output))
                ],
                source_sentence=sentence,
                src=self.src,
                tgt=self.tgt,
                strategy="greedy",
            )
            for sentence, output in zip(sentences, outputs)
        ]

        return translations

    def cot(
        self,
        prompts: List[str],
        sentences: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ):
        prompts = [self.apply_chat_template(prompt) for prompt in prompts]
        outputs = []
        for i in range(0, len(prompts), request_batch_size):
            output = self.generate(
                prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                outputs.append(output[j])

        # Answer extraction
        extraction_prompts = []
        for i, _ in enumerate(prompts):
            for j in range(len(outputs[i])):
                # outputs is a List[List[str]]
                extraction_prompt = f"Here is a reasoning about the translation of a sentence written in {self.src} into a sentence written in {self.tgt}."
                extraction_prompt += (
                    f"\n\n<Reasoning>\n\n{outputs[i][j].strip()}\n</Reasoning>"
                )
                extraction_prompt += f"\n\nExtract the final {self.tgt} translation from the reasoning and write it down. Don't write anything else except the final translation extracted from the reasoning above."
                extraction_prompt += (
                    f"Don't use the additional/superfluous quotation mark or bolding."
                )
                extraction_prompts.append(extraction_prompt)

        extraction_prompts = [
            self.apply_chat_template(prompt) for prompt in extraction_prompts
        ]
        assert len(extraction_prompts) == num_return_sequences * len(
            prompts
        ), "There is a size mismatch in Zero-shot CoT"
        final_outputs = []
        for i in range(0, len(extraction_prompts), request_batch_size):
            # Answer extraction does not require sampling
            output = self.generate(
                extraction_prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,  # 0.0
                num_return_sequences=1,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                final_outputs.extend(output[j])

        final_outputs = [output.strip().split("\n")[0] for output in final_outputs]
        final_outputs = [
            _stop_at_stop_token(output, STOP_WORDS) for output in final_outputs
        ]
        final_outputs = [remove_repeating_bigram(output) for output in final_outputs]
        outputs_list = []

        for i in range(0, len(final_outputs), num_return_sequences):
            outputs_list.append(
                get_best_sentence(
                    target_translations=final_outputs[i : i + num_return_sequences],
                    source_sentence=sentences[i // num_return_sequences],
                    src=self.src,
                    tgt=self.tgt,
                )
            )
        return outputs_list

    def maps(
        self,
        sentences: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[str]:
        demos_prompts = [
            get_maps_aspects(sentence, self.src, self.tgt, "demos")
            for sentence in sentences
        ]
        keywords_prompts = [
            get_maps_aspects(sentence, self.src, self.tgt, "keywords")
            for sentence in sentences
        ]
        topics_prompts = [
            get_maps_aspects(sentence, self.src, self.tgt, "topics")
            for sentence in sentences
        ]
        demos_outputs = []
        keywords_outputs = []
        topics_outputs = []

        for i in range(0, len(demos_prompts), request_batch_size):
            # Demonstrations
            output = self.generate(
                demos_prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                demos_outputs.extend(output[j])
            # Keywords
            output = self.generate(
                keywords_prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                keywords_outputs.extend(output[j])
            # Topics
            output = self.generate(
                topics_prompts[i : i + request_batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                topics_outputs.extend(output[j])

        demos_outputs = [output.strip().split("\n\n")[0] for output in demos_outputs]
        keywords_outputs = [
            output.strip().split("\n\n")[0] for output in keywords_outputs
        ]
        topics_outputs = [output.strip().split("\n\n")[0] for output in topics_outputs]
        if verbose:
            for i, (a, b, c) in enumerate(
                zip(demos_outputs, keywords_outputs, topics_outputs)
            ):
                print(f"=> {i+1}. Demonstrations:\n{a}\nKeywords:\n{b}\nTopics:\n{c}\n")
        # Translation
        zs_prompts = [
            get_maps_aspects(sentence, self.src, self.tgt, "trans-zs")
            for sentence in sentences
        ]
        demos_prompts = [
            get_maps_aspects(sentence, self.src, self.tgt, "trans-demos", demos=demos)
            for sentence, demos in zip(sentences, demos_outputs)
        ]
        keywords_prompts = [
            get_maps_aspects(
                sentence, self.src, self.tgt, "trans-keywords", keywords=keywords
            )
            for sentence, keywords in zip(sentences, keywords_outputs)
        ]
        topics_prompts = [
            get_maps_aspects(
                sentence, self.src, self.tgt, "trans-topics", topics=topics
            )
            for sentence, topics in zip(sentences, topics_outputs)
        ]

        zs_prompts = [self.apply_chat_template(prompt) for prompt in zs_prompts]
        demos_prompts = [self.apply_chat_template(prompt) for prompt in demos_prompts]
        keywords_prompts = [
            self.apply_chat_template(prompt) for prompt in keywords_prompts
        ]
        topics_prompts = [self.apply_chat_template(prompt) for prompt in topics_prompts]

        # Translations
        translations = [[] for _ in range(len(sentences))]
        for _, prompts in enumerate(
            [zs_prompts, demos_prompts, keywords_prompts, topics_prompts]
        ):
            for i in range(0, len(prompts), request_batch_size):
                output = self.generate(
                    prompts[i : i + request_batch_size],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    top_p=top_p,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    request_batch_size=request_batch_size,
                    verbose=verbose,
                )
                for j in range(len(output)):
                    translations[i + j].extend(output[j])
        if verbose:
            print(f"A: {translations[0]}")
            print(f"B: {translations[-1]}")
        sources = [[sentences[i]] * len(translations[i]) for i in range(len(sentences))]
        translations = [
            [element.strip().split("\n")[0] for element in translation]
            for translation in translations
        ]
        assert len(sources) == len(translations)
        # Flatten
        sources = [x for xs in sources for x in xs]
        predictions = [x for xs in translations for x in xs]
        scores = quality_estimation(sources, predictions)
        # Selection
        outputs_list = []
        current = 0
        for i in range(len(sentences)):
            outputs_list.append(
                translations[i][
                    np.argmax(scores[current : current + len(translations[i])])
                ]
            )
            if verbose:
                prompt = "===\nThe best sentence between the following\n"
                for j in range(len(translations[i])):
                    prompt += f"{j+1}. {translations[i][j]}\n"
                prompt += f"Is << {outputs_list[-1]} >>\n==="
                print(prompt)
            current += len(translations[i])

        return outputs_list

    def step_by_step(
        self,
        sentences: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[str]:
        final_outputs = []
        for i in range(0, len(sentences), request_batch_size):
            inputs = sentences[i : i + request_batch_size]
            pre_translation_prompts = [
                get_step_by_step_prompts(
                    description="pre-translation-research",
                    src=self.src,
                    tgt=self.tgt,
                    source=sentence,
                )
                for sentence in inputs
            ]
            processed_pre_translation_prompts = [
                self.apply_chat_template(prompt) for prompt in pre_translation_prompts
            ]
            # print(f"---\nFIRST PRE-TRANSLATION PROMPT\n---\n{processed_pre_translation_prompts[0]}\n===")
            pre_translation_outputs = []
            output = self.generate(
                processed_pre_translation_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                pre_translation_outputs.extend(output[j])
            # draft
            draft_prompts = [
                [
                    {"role": "user", "content": pre_translation_prompts[j]},
                    {
                        "role": "assistant",
                        "content": pre_translation_outputs[j].strip(),
                    },
                    {
                        "role": "user",
                        "content": get_step_by_step_prompts(
                            description="drafting", src=self.src, source=inputs[j]
                        ),
                    },
                ]
                for j in range(len(inputs))
            ]
            processed_draft_prompts = [
                self.apply_chat_template(prompt) for prompt in draft_prompts
            ]
            # print(f"---\nFIRST DRAFT PROMPT\n---\n{processed_draft_prompts[0]}\n===")
            draft_outputs = []
            output = self.generate(
                processed_draft_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                draft_outputs.extend(output[j])
            draft_outputs = [element.strip() for element in draft_outputs]
            if verbose:
                for a, b in zip(inputs, draft_outputs):
                    print("-" * 70 + f"\n{self.src}: {a}\n{self.tgt}: {b}\n" + "-" * 70)
                    break
            # refinement
            refine_prompts = [
                draft_prompts[j]
                + [
                    {"role": "assistant", "content": draft_outputs[j]},
                    {
                        "role": "user",
                        "content": get_step_by_step_prompts(description="refinement"),
                    },
                ]
                for j in range(len(inputs))
            ]
            processed_refine_prompts = [
                self.apply_chat_template(prompt) for prompt in refine_prompts
            ]
            # print(f"---\nFIRST REFINE PROMPT\n---\n{processed_refine_prompts[0]}\n===")
            refine_outputs = []
            output = self.generate(
                processed_refine_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                refine_outputs.extend(output[j])
            refine_outputs = [element.strip() for element in refine_outputs]
            if verbose:
                for a, b in zip(inputs, refine_outputs):
                    print("-" * 70 + f"\n{self.src}: {a}\n{self.tgt}: {b}\n" + "-" * 70)
                    break
            # Proof reading, new conversation
            proofreading_prompts = [
                get_step_by_step_prompts(
                    description="proofreading",
                    source=inputs[j],
                    draft=draft_outputs[j],
                    refine=refine_outputs[j],
                )
                for j in range(len(inputs))
            ]
            processed_proofreading_prompts = [
                self.apply_chat_template(prompt) for prompt in proofreading_prompts
            ]
            # print(f"---\nFIRST PROOFREADING PROMPT\n---\n{processed_proofreading_prompts[0]}\n===")
            outputs = []
            output = self.generate(
                processed_proofreading_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                request_batch_size=request_batch_size,
                verbose=verbose,
            )
            for j in range(len(output)):
                outputs.extend(output[j])
            if verbose:
                for a, b in zip(inputs, outputs):
                    print("-" * 70 + f"\n{self.src}: {a}\n{self.tgt}: {b}\n" + "-" * 70)
                    break
            outputs = [output.strip().split("\n")[0] for output in outputs]
            outputs = [remove_repeating_bigram(output) for output in outputs]
            final_outputs.extend(outputs)
        return final_outputs

    def tear(
        self,
        sentences: List[str],
        demonstrations: List[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[str]:
        final_outputs = []
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_return_sequences": 1,
            "top_p": top_p,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "request_batch_size": request_batch_size,
            "verbose": verbose,
        }
        for i in range(0, len(sentences), request_batch_size):
            inputs = sentences[i : i + request_batch_size]
            translate_prompts = [
                get_tear_prompts(
                    description="translate",
                    src=self.src,
                    tgt=self.tgt,
                    source=sentence,
                    demonstrations=demonstrations[j],
                )
                for j, sentence in enumerate(inputs)
            ]
            translate_prompts = [
                self.apply_chat_template(prompt) for prompt in translate_prompts
            ]
            print(f"<<<TRANSLATE>>>\n{translate_prompts[0]}")
            translate_outputs = []
            output = self.generate(translate_prompts, **generation_kwargs)
            for j in range(len(output)):
                translate_outputs.extend(output[j])
            if verbose:
                for a, b in zip(inputs, translate_outputs):
                    print(f"{self.src}: {a}\n{self.tgt}: {b}")
                    break
            translate_outputs = [
                output.strip().split("\n")[0].strip() for output in translate_outputs
            ]
            estimate_prompts = [
                get_tear_prompts(
                    description="estimate",
                    src=self.src,
                    tgt=self.tgt,
                    source=inputs[j],
                    draft=translate_outputs[j],
                )
                for j in range(len(inputs))
            ]
            estimate_prompts = [
                self.apply_chat_template(prompt) for prompt in estimate_prompts
            ]
            print(f"<<<ESTIMATE>>>\n{estimate_prompts[0]}")
            estimate_outputs = []
            output = self.generate(estimate_prompts, **generation_kwargs,)
            if verbose:
                for a, b in zip(inputs, estimate_outputs):
                    print(f"{self.src}: {a}\n{self.tgt}: {b}")
                    break
            for j in range(len(output)):
                estimate_outputs.extend(output[j])
            estimate_outputs = [output.strip() for output in estimate_outputs]
            refine_prompts = [
                get_tear_prompts(
                    description="refine",
                    src=self.src,
                    tgt=self.tgt,
                    source=inputs[j],
                    draft=translate_outputs[j],
                    demonstrations=demonstrations[j],
                    estimate_fdb=estimate_outputs[j].strip(),
                )
                for j in range(len(inputs))
            ]
            refine_prompts = [
                self.apply_chat_template(prompt) for prompt in refine_prompts
            ]
            print(f"<<<REFINE>>>\n{refine_prompts[0]}")
            refine_outputs = []
            output = self.generate(refine_prompts, **generation_kwargs)
            for j in range(len(output)):
                refine_outputs.extend(output[j])
            outputs = [
                output.strip().split("\n")[0].strip() for output in refine_outputs
            ]
            final_outputs.extend(outputs)

        return final_outputs


import cohere


class cohereSampler(Sampler):
    def __init__(self, api_key=None, max_retry=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry = max_retry
        self.api_key = (
            api_key
            if api_key
            else os.environ.get("COHERE_API_KEY", "<YOUR_COHERE_API_KEY>")
        )
        self.client = cohere.Client(self.api_key)

    def generate(
        self,
        # prompts: List[str],
        prompts,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            prompts = [prompts]
        attempt = 0
        responses = []
        while attempt < self.max_retry:
            try:
                start = len(responses)
                for q, prompt in enumerate(prompts):
                    if q < start:
                        continue
                    response = self.client.chat(
                        model=self.model_name_or_path,
                        chat_history=[
                            {"role": "SYSTEM", "message": "You are a helpful assistant"}
                        ]
                        if isinstance(prompt, str)
                        else prompt[:-1],
                        message=prompt
                        if isinstance(prompt, str)
                        else prompt[-1]["message"],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        p=top_p,
                    )
                    responses.append(response)
                break
            except Exception as e:
                print(f"CohereError: {e}.")
                attempt += 1
        assert len(responses) == len(
            prompts
        ), f"Size mismatch between {len(responses)} and {len(prompts)}."
        outputs = []
        for j, response in enumerate(responses):
            if verbose:
                # print(f"{j+1} -> {response.text}\n{response.finish_reason}")
                print(f"{j+1} -> {response.text}\n")
            outputs.append([response.text])
        return outputs


from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch


class HFSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                self.tokenizer_name_or_path
                if self.tokenizer_name_or_path
                else self.model_name_or_path
            ),
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
            if (self.tokenizer.pad_token is None)
            else self.tokenizer.pad_token
        )
        self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map={"": self.accelerator.process_index},
            # torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            trust_remote_code=True,
            attn_implementation=(
                "eager"
                if "gemma-2-" in self.model_name_or_path
                else "flash_attention_2"
            ),
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        response = hf_generate(
            accelerator=self.accelerator,
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_words=[
                "\n###",
                "<|eot_id|>",
                "\nHuman:",
                "<end_of_turn>",
                "<|start_header_id|>",
            ],
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            forced_bos_token_id=None,
            batch_size=request_batch_size,
        )
        outputs = []
        for i, r in enumerate(response):
            output = r["answer"]
            outputs.append(output)
        if verbose:
            print("===")
            for i, output in enumerate(outputs):
                for out in output:
                    print(f"{i+1} -> {out}")
            print("===")
        return outputs


import os
import openai


class OpenAISampler(Sampler):
    def __init__(self, api_key, max_retry=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retry = max_retry
        self.api_key = api_key if api_key else os.environ["OPENAI_API_KEY"]
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        if self.model_name_or_path in [
            "babbage-002",
            "davinci-002",
            "gpt-3.5-turbo-instruct",
        ]:
            attempt = 0
            while attempt < self.max_retry:
                try:
                    response = self.client.completions.create(
                        model=self.model_name_or_path,
                        prompt=prompts,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        best_of=num_beams,
                    )
                    break
                except Exception as e:
                    print(f"OpenAIError: {e}.")
                    attempt += 1
            outputs = []
            for j, _ in enumerate(prompts):
                output = response.choices[
                    j * num_return_sequences : (j + 1) * num_return_sequences
                ]
                if verbose:
                    for _, out in enumerate(output):
                        print(f"{j+1} -> {out.text}\n{out.finish_reason}")
                outputs.append([out.text for out in output])
            return outputs
        else:
            attempt = 0
            responses = []
            while attempt < self.max_retry:
                try:
                    start = len(responses)
                    for q, prompt in enumerate(prompts):
                        if q < start:
                            continue
                        response = self.client.chat.completions.create(
                            model=self.model_name_or_path,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_new_tokens,
                            n=num_return_sequences,
                            # seed=self.seed
                        )
                        responses.append(response)
                    break
                except Exception as e:
                    print(f"OpenAIError: {e}.")
                    attempt += 1
            assert len(responses) == len(prompts), "Size mismatch."
            outputs = []
            for j, response in enumerate(responses):
                if verbose:
                    for choice in response.choices:
                        print(
                            f"{j+1} -> {choice.message.content}\n{choice.finish_reason}"
                        )
                outputs.append([choice.message.content for choice in response.choices])
            return outputs


import anthropic


class AnthropicSampler(Sampler):
    def __init__(self, api_key, max_retry=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = (
            api_key
            if api_key
            else os.environ.get("ANTHROPIC_API_KEY", "YOU_ANTHROPIC_API_KEY")
        )
        self.max_retry = max_retry
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: int,
        num_return_sequences: int,
        num_beams: int,
        do_sample: bool,
        request_batch_size: int,
        verbose: bool = True,
    ) -> List[List[str]]:
        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        attempt = 0
        responses = []
        while attempt < self.max_retry:
            try:
                start = len(responses)
                for q, prompt in enumerate(prompts):
                    if q < start:
                        continue
                    response = self.client.messages.create(
                        model=self.model_name_or_path,
                        # system="You are a highly skilled translator with expertise in many languages. Your task is to identify the language of the text I provide and accurately translate it into the specified target language while preserving the meaning, tone, and nuance of the original text. Please maintain proper grammar, spelling, and punctuation in the translated version.",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        stop_sequences=["\n###"],
                        top_p=top_p,
                    )
                    responses.append(response)
                break
            except Exception as e:
                print(f"AnthropicError: {e}.")
                attempt += 1
        assert len(responses) == len(prompts), "Size mismatch."
        outputs = []
        for j, response in enumerate(responses):
            if verbose:
                for content in response.content:
                    print(f"{j+1} -> {content.text}\n{response.stop_reason}")
            outputs.append([choice.text for choice in response.content])
        return outputs


from vllm import LLM, SamplingParams


class vLLMSampler(Sampler):
    def __init__(self, enable_lora=False, max_lora_rank=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            best_of=1,
            repetition_penalty=1.03,
            use_beam_search=False,
            skip_special_tokens=True,
            stop=[
                "\n###",
                "###",
                "://",
                "<|eot_id|>",
                "\nHuman:",
                "\n<end_of_turn>",
                "<|start_header_id|>",
                "\n\n\n\n\n",
                ">>>>>",
                "```python",
                "=====",
                "\\\n" * 4,
                "<EOS_TOKEN>",
                "\n\n\n\n",
            ],
        )
        if "gemma-2-" in self.model_name_or_path:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        if (
            "awq" in self.model_name_or_path.lower()
            or "gptq" in self.model_name_or_path.lower()
        ):
            self.llm = LLM(
                model=self.model_name_or_path,
                tokenizer=self.tokenizer_name_or_path,
                quantization=(
                    "AWQ" if "awq" in self.model_name_or_path.lower() else "GPTQ"
                ),
                dtype="half",
                max_model_len=(
                    2048
                    if any(
                        [
                            element in self.model_name_or_path
                            for element in ["bloom", "OLMo", "opt", "xglm", "llama-2"]
                        ]
                    )
                    # else 4096
                    # else 3584
                    else 3072
                ),
                enforce_eager=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                enable_lora=enable_lora,
                max_lora_rank=max_lora_rank,
            )
        else:
            self.llm = LLM(
                model=self.model_name_or_path,
                tokenizer=self.tokenizer_name_or_path,
                dtype=("bfloat16" if "gemma-2-" in self.model_name_or_path else "half"),
                max_model_len=(
                    2048
                    if any(
                        [
                            element in self.model_name_or_path
                            for element in ["bloom", "OLMo", "opt", "xglm"]
                        ]
                    )
                    else 4096
                ),
                enforce_eager=True,
                trust_remote_code=True,
                swap_space=8,
                enable_lora=enable_lora,
                max_lora_rank=max_lora_rank,
                tensor_parallel_size=torch.cuda.device_count()
            )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        num_beams: int = 1,
        do_sample: bool = False,
        request_batch_size: int = 8,
        verbose: bool = True,
    ) -> List[List[str]]:
        # Initialization
        self.sampling_params.temperature = temperature
        self.sampling_params.top_p = top_p
        self.sampling_params.best_of = num_beams
        self.sampling_params.repetition_penalty = repetition_penalty
        self.sampling_params.use_beam_search = not do_sample and num_beams > 1
        self.sampling_params.max_tokens = max_new_tokens
        self.sampling_params.n = num_return_sequences
        self.sampling_params.skip_special_tokens = True
        self.sampling_params.ignore_eos = False

        if isinstance(prompts, list):
            pass
        else:
            # single prompt, i.e str
            prompts = [prompts]
        response = self.llm.generate(prompts, self.sampling_params)
        if verbose:
            print("===")
            for i, r in enumerate(response):
                for element in r.outputs:
                    print(f"{i+1} -> {element.text}\n{element.finish_reason}")
            print("===")
        return [[element.text for element in r.outputs] for r in response]


if __name__ == "__main__":
    sampler = cohereSampler(
        model_name_or_path="command-r-08-2024",
        tokenizer_name_or_path=None,
        src="English",
        tgt="French",
        template=None,
        merge_prompt="vanilla",
        # method_translate="vanilla",
        method_translate="nllb",
        selection_method="comet-qe",
        nllb_name_or_path="facebook/nllb-200-distilled-600M",
    )
    sentence = "I am very inclined towards eating your pancreas to alleviate my hunger."
    t = sampler.translate(
        sentences=[sentence], temperature=0.7, num_return_sequences=4, do_sample=False
    )
    print(f"NLLB: {t[0]}")
    """
    outputs = sampler.generate(
        prompts=[
            f"What is the French translation of the following sentence\n\n{sentence}"
        ],
        max_new_tokens=100,
        top_p=1.0,
        temperature=0,
    )
    print(f"zero-shot MT: {outputs[0]}")
    outputs = sampler.cot(
        prompts=[
            f"What is the French translation of the following sentence\n\n{sentence}\n\nLet's think step by step."
        ],
        sentences=[sentence],
        max_new_tokens=500,
    )
    print(f"zero-shot CoT MT: {outputs[0]}")
    outputs = sampler.maps(sentences=[sentence], max_new_tokens=500)
    print(f"MAPS MT: {outputs[0]}")
    
    sentences = [sentence]
    subsentences = sampler.divide(n_splits=-1, sentences=sentences)
    print(subsentences)
    translations = []
    for i in range(len(subsentences)):
        t = sampler.translate(
            sentences=subsentences[i],
            demonstrations=[[] for _ in range(len(subsentences[i]))],
        )
        translations.append(t)
    print(translations)
    comptra_translations = sampler.merge(
        sentences=sentences,
        inputs=subsentences,
        outputs=translations,
        max_new_tokens=300,
    )
    print(f"CoTra translation: {comptra_translations[0]}")
    refine_translations = sampler.refine(
        sentences = sentences,
        prev_translations = comptra_translations
    )
    """
