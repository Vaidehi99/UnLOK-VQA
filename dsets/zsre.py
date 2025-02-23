import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size: int = None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        # get all possible answers
        all_answers = []
        for i, record in enumerate(raw):
            all_answers.append(record['answers'][0])

        data = []
        print("Loading zsre data...")
        for i, record in enumerate(raw):
            if 'subject' not in record or 'neigh_idx' not in record:
                continue
            if len(record["loc"])==0:
                continue
            print(f"loading point {i}", end='\r')
            assert (
                "nq question: " in record["loc"][0]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = record["loc_ans"] #[tok(" " + x)["input_ids"] for x in record["loc_ans"]]
            # ans_toks = tok(record["loc_ans"])["input_ids"]

            true_answer = record["answers"][0]
            eligible_alt_answers = np.setdiff1d(all_answers, true_answer)
            # adjust prompt formatting a bit -- occurs for paraphrases later too
            question = record["src"].replace(record["subject"], "{}")
            prompt = f"Answer the question in one word\n Question: {question} Answer:"
            k=0
            if len(record["alt_ans"])>0:
                for k in range(len(record["alt_ans"])):
                    if len(record["alt_ans"][k].strip())>0: 
                        alt_ans = record["alt_ans"][k]
                        break
            else:
                alt_ans = np.random.choice(eligible_alt_answers)
            if len(alt_ans)<=0:
                alt_ans = np.random.choice(eligible_alt_answers)

            if len(record["objects_edit"])>0:
                 objects_edit = record["objects_edit"][0]
            else:
                objects_edit = record["subject"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "id": i,
                        "prompt": prompt,
                        "subject": record["subject"],
                        "image_id": record["image_id"],
                        "objects_edit": objects_edit,
                        # ROME paper uses editing to correct mistakes, unlike the CounterFact setting
                        # "target_new": {"str": record["answers"][0]},
                        # "target_true": {"str": "<|endoftext|>"},
                        # even though error fixing is a better eval, we compatabilize with CounterFact
                        # by setting the new target to a random entity
                        "target_new": {"str": alt_ans},
                        "target_true": {"str": true_answer},
                        "target_empty": {"str": "I don't know"},  
                    },
                    "alt_ans": record["alt_ans"],
                    "neigh_idx": record["neigh_idx"],                    
                    "paraphrase_prompts": [f"Answer the question in one word\n Question: {record['rephrase'][i]} Answer:" for i in range(len(record['rephrase']))],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"][i],
                            "target": ans_toks[i].split(" ")[0],
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )
            if size is not None and i == size:
                break

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
