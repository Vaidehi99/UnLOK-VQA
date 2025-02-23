import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional
import sys
sys.path.append("/nas-ssd2/vaidehi/nlp13/belief-localization/third_party/")

from util.fewshot_utils import score_from_batch



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def compute_loss(self, model, inputs):
        print(self.args.defense)
        exit()
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.defense=="fact_erasure":
            print("rur")
            return -loss
        else:
            return loss

    def compute_loss(self, model, inputs):
        print(self.args.defense)
        # print(inputs)
        # print(len(inputs['input_ids']))
        # print(inputs['input_ids'][0].shape)
        # print(len(inputs['images']))
        # print(inputs['images'][0].shape)
        # print(len(['target_ids']))
        # print(inputs['target_ids'][0].shape)
   
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        seq_log_probs = score_from_batch(model, inputs, return_log_probs=True)
        nll = -seq_log_probs.sum()
        pred_prob = torch.exp(-nll)

        if self.args.defense=="fact_erasure":
            if self.args.margin_loss:
                margin_layers = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                model_batch = {}
                for key in ["input_ids", "attention_mask"]:
                    model_batch[key] = inputs[key][:,:-1]
                model_batch["images"] = inputs["images"]
                k_new = 100

                hidden_states = model(**model_batch, output_hidden_states=True).hidden_states
                hidden_states = [hid[:, -model_batch["input_ids"].shape[1]:, :] for hid in hidden_states]
                layer_logits = collect_logits(model, hidden_states, layers=margin_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(model_batch["attention_mask"], -1)-1, model_batch["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
                sorted_layer_logits, _ = torch.sort(layer_logits, -1)
                min_topk_prob = sorted_layer_logits[:,:,-(k_new)]
                max_bottomk_prob = sorted_layer_logits[:,:,k_new-1]
                # print(inputs["target_ids"][0][-1])
                # print(layer_logits.shape)
                # print(layer_logits[:,:,inputs["target_ids"][0][-1]])
                target_prob = layer_logits[:,:,inputs["target_ids"][0][-1]]
                margin_top = torch.maximum(torch.zeros_like(target_prob), target_prob-min_topk_prob)#*input_tok["attention_mask"].expand(target_prob.shape[0], -1, -1)
                margin_bottom = torch.maximum(torch.zeros_like(target_prob), max_bottomk_prob-target_prob)#*input_tok["attention_mask"].expand(target_prob.shape[0], -1, -1)
                margin_loss_top = margin_top.mean()
                margin_loss_bottom = margin_bottom.mean()
                loss = margin_loss_top + margin_loss_bottom
                print(f"Loss: {loss} = {margin_loss_top} + {margin_loss_bottom}")
                return loss

            elif self.args.entropy_loss:
                entropy_layers = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                # hidden_states = model(**inputs, output_hidden_states=True).hidden_states
                # hidden_states = [hid[:, -inputs["input_ids"].shape[1]:, :] for hid in hidden_states]
                # layer_logits = collect_logits(model, hidden_states, layers=self.args.margin_layers)
                # last_nonzero_mask = torch.remainder(torch.argmin(inputs["attention_mask"], -1)-1, inputs["attention_mask"].shape[-1])
                # target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                # expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                # layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
                model_batch = {}
                for key in ["input_ids", "attention_mask"]:
                    model_batch[key] = inputs[key][:,:-1]
                model_batch["images"] = inputs["images"]
                k_new = 100

                hidden_states = model(**model_batch, output_hidden_states=True).hidden_states
                hidden_states = [hid[:, -model_batch["input_ids"].shape[1]:, :] for hid in hidden_states]
                layer_logits = collect_logits(model, hidden_states, layers=entropy_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(model_batch["attention_mask"], -1)-1, model_batch["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)

                entropy = Categorical(probs = layer_logits.float()).entropy().mean()
                loss = -entropy
                print(f"Loss: {loss}")
                return loss

            else:
                print(f"Loss: {pred_prob}")
                # for (n, p) in model.named_parameters():
                #     if p.requires_grad:
                #         print(n)
                # print("x")
                # import pdb; pdb.set_trace();

                return pred_prob
        if self.args.defense=="error_injection" or self.args.defense=="empty_response":
            print(f"Loss: {nll}")
            return nll
        
        # print(x)
        # outputs = model(**inputs)

        # print(outputs.logits.shape)
        # exit()
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.defense=="fact_erasure":
            print(f"Loss: {loss}")
            return -loss
        else:
            print(f"Loss: {loss}")
            return loss
        
    def compute_loss_fact_erasure_works(self, model, inputs):
        print(self.args.defense)

   
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.defense=="fact_erasure":
            print(f"Loss: {loss}")
            return -loss
        else:
            print(f"Loss: {loss}")
            return loss        



        



    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
