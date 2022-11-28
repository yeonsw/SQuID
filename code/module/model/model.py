import os

from accelerate import Accelerator
import apex
from dataclasses import dataclass, field
import logging
import math
import nltk
import numpy as np
import random
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import ( 
    BertTokenizer, 
    BertModel, 
    BertPreTrainedModel,
    BertForSequenceClassification,
    AlbertPreTrainedModel,
    AlbertModel,
    DPRQuestionEncoder,
    DPRConfig,
    DPRPretrainedQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    AutoModel,
    AdamW, 
    get_scheduler,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from typing import Optional, Union

from .tokenizer import get_tokenizer
from .train_data import DPRDataset, data_collator
from .pred_data import DPRPredDataset, pred_data_collator
from .pred_data_inf_speed import DPRPredInfSpeedDataset, pred_data_inf_speed_collator
from .metric import compute_f1_sets, compute_exact_sets

import time

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    init_checkpoint: Optional[str] = field(default=None)
    checkpoint_save_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    per_device_inds_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    eval_steps: Optional[int] = field(default=16)
    n_epochs: Optional[int] = field(default=3)
    weight_decay: Optional[float] = field(default=0.01)
    learning_rate: Optional[float] = field(default=2e-5)
    warmup_ratio: Optional[float] = field(default=0.1)

@dataclass
class PredModelArguments:
    init_checkpoint: Optional[str] = field(default=None)
    per_device_eval_batch_size: Optional[int] = field(default=16)

class ALBERTForQsim(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.albert = AlbertModel(config)
        self.qsim_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.qsim_layer(pooled_output)
        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=logits,
            hidden_states=None,
            attentions=None,
        )

class BERTForQsim(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.qsim_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.qsim_layer(pooled_output)
        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=logits,
            hidden_states=None,
            attentions=None,
        )

class DPRQsimTrainer:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.accelerator = Accelerator()
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint \
            )
        self.model = self.get_model()
        self.model = self.model.to(self.device) 

        self.train_dataloader = None
        self.eval_dataloader = None
        assert self.data_config.train_file != None \
            and self.data_config.eval_file != None
        self.train_dataset, \
            self.train_dataloader = \
                self.get_train_data( \
                    self.data_config.train_file, \
                )
        
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_eval_data( \
                    self.data_config.eval_file, \
                )
        
        self.optimizer, \
            self.lr_scheduler, \
            self.total_steps = \
            self.get_optimizer()

        apex.amp.register_half_function(torch, 'einsum')
        self.model, self.optimizer \
            = apex.amp.initialize(
                self.model, self.optimizer, \
                opt_level="O1")
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_eval_data(self, input_file):
        dpr_dataset = DPRPredDataset( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dpr_dataloader = DataLoader(
            dpr_dataset,
            shuffle=False,
            collate_fn=pred_data_collator,
            batch_size= \
                self.model_config.per_device_eval_batch_size * self.n_gpus
        )
        return (dpr_dataset, dpr_dataloader)
    
    def get_train_data(self, input_file):
        dpr_dataset = DPRDataset( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dpr_dataloader = DataLoader(
            dpr_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size= \
                self.model_config.per_device_train_batch_size * self.n_gpus \
        )
        return (dpr_dataset, dpr_dataloader)

    def get_model(self):
        if self.model_config.init_checkpoint.startswith("bert"):
            MODEL_CLASS = BERTForQsim
        elif self.model_config.init_checkpoint.startswith("albert"):
            MODEL_CLASS = ALBERTForQsim
        else:
            MODEL_CLASS = DPRQuestionEncoder

        model = MODEL_CLASS.from_pretrained(
                self.model_config.init_checkpoint,
        )
        return model
    
    def get_optimizer(self):
        assert self.train_dataloader != None
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if not any(nd in n for nd in no_decay) \
                ],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [ \
                    p for n, p in self.model.named_parameters() \
                        if any(nd in n for nd in no_decay) \
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW( \
            optimizer_grouped_parameters, \
            lr=self.model_config.learning_rate \
        )
       
        n_steps_per_epoch = \
            math.ceil( \
                len(self.train_dataloader)\
                  / self.model_config.gradient_accumulation_steps \
            )
        train_steps = self.model_config.n_epochs \
                        * n_steps_per_epoch
        n_warmup_steps = int( \
            train_steps * self.model_config.warmup_ratio \
        )
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=n_warmup_steps,
            num_training_steps=train_steps,
        )
        return (optimizer, lr_scheduler, train_steps)
    
    def get_loss(self, q_vecs, n_samples):
        total_n, _ = q_vecs.size()
        batch_mask_indices = torch.cat((torch.eye(n_samples), torch.zeros(n_samples, total_n - n_samples)), dim=1).to(q_vecs.device)
        pos_indices = torch.arange(n_samples, 2 * n_samples).to(q_vecs.device)
        
        input_q_vecs, _ = \
            torch.split( \
                q_vecs, \
                [n_samples, total_n - n_samples], \
                dim=0 \
            )
        loss_fct = nn.CrossEntropyLoss()
        sim = torch.matmul( \
            input_q_vecs, \
            q_vecs.transpose(0, 1) \
        )
        sim = sim.masked_fill(batch_mask_indices.bool(), float('-inf'))
        loss = loss_fct(sim, pos_indices)
        return (loss, sim)

    def train_model(self):
        total_batch_size = \
            self.model_config \
                .per_device_train_batch_size \
            * self.accelerator.num_processes \
            * self.model_config.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.model_config.n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.model_config.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.model_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.total_steps}")
        
        last_checkpoint_dir = "{}_{}".format(self.model_config.checkpoint_save_dir.rstrip("/"), "last_checkpoint")
        total_train_loss = 0.0
        global_steps = 0
        best_eval_score = float("-inf")
        self.model.zero_grad()
        for i in range(0, self.model_config.n_epochs):
            self.model.train()
            progress_bar = tqdm( \
                self.train_dataloader, \
                desc="Training ({:d}'th iter / {:d})".format(i+1, self.model_config.n_epochs, 0.0, 0.0) \
            )
            
            for step, batch in enumerate(progress_bar):
                self.model.train()
                target_inputs = [ \
                    "input_ids", "token_type_ids", "attention_mask"
                ]
                new_batch = {k: batch[k].to(self.device) for k in target_inputs}
                q_vecs = self.model(**new_batch).pooler_output
                output = self.get_loss( \
                    q_vecs, \
                    batch["n_samples"]
                )
                loss = output[0]
                loss = loss / self.model_config.gradient_accumulation_steps
                
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                
                if step % self.model_config.gradient_accumulation_steps == 0 \
                    or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_steps += 1
                total_train_loss += loss.item()
                
                if global_steps % self.model_config.eval_steps == 1:
                    avg_train_loss = \
                        total_train_loss \
                            / self.model_config.eval_steps
                    val_em, val_f1 = self.eval_model()
                    if val_f1 > best_eval_score:
                        best_eval_score = val_f1
                        self.save_model(self.model_config.checkpoint_save_dir)
                    self.save_model(last_checkpoint_dir)
                    desc_template = "Training ({:d}'th iter / {:d})|loss: {:.03f}, Val: {:.03f}/{:.03f}"
                    print('\n')
                    print(desc_template.format( \
                            i+1, \
                            self.model_config.n_epochs, \
                            avg_train_loss, \
                            val_em, \
                            val_f1, \
                        )
                    )
                    total_train_loss = 0.0

        val_em, val_f1 = self.eval_model()
        desc_template = "Training (Final) | Val: {:.03f}/{:.03f}"
        print('\n')
        print(desc_template.format( \
                val_em, \
                val_f1, \
            )
        )
        if val_f1 > best_eval_score:
            best_eval_score = val_f1
            self.save_model(self.model_config.checkpoint_save_dir)
        self.save_model(last_checkpoint_dir)
        return 0
    
    def eval_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", "attention_mask"
        ]
        em = []
        f1 = []
        self.model.eval()
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                bsize = batch["n_samples"]
                questions = batch["questions"]
                bm25_questions = batch["bm25_questions"]
                gt_answers = batch["gt_answer"]
                bm25_answers = batch["bm25_answers"]
                
                new_batch = {k: batch[k].to(self.device) for k in target_inputs}
                q_vecs = self.model(**new_batch) \
                    .pooler_output.detach().cpu().numpy()
                n_vecs, _ = q_vecs.shape
                queries = q_vecs[:bsize]
                
                chunk_size = batch["bm25_size"][0]
                n_chunks = (n_vecs - bsize) // chunk_size
                assert n_chunks * chunk_size == (n_vecs - bsize)
                bm25_vecs = np.split(q_vecs[bsize:], n_chunks)
                
                for question, q, cand, gt_answer, bm25_qs, bm25_answer in zip(questions, queries, bm25_vecs, gt_answers, bm25_questions, bm25_answers):
                    sims = np.dot(cand, q)
                    pred_idx = np.argmax(sims)
                    
                    topk = np.argsort(-sims)[:1] #top K
                    pred_answer = []
                    for m_idx in topk:
                        pred_answer += bm25_answer[m_idx]
                    
                    f1.append(compute_f1_sets(set(gt_answer), set(pred_answer)))
                    em.append(compute_exact_sets(set(gt_answer), set(pred_answer)))
        em = 100 * np.mean(em)
        f1 = 100 * np.mean(f1)
        return (em, f1)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        model_to_save = \
            self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("\nSaving model checkpoint to %s", output_dir)

        return 0

class DPRQsimPred:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.accelerator = Accelerator()
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint \
            )
        self.model = self.get_model()
        self.model = self.model.to(self.device) 

        self.eval_dataloader = None
        assert self.data_config.pred_file != None
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.pred_file, \
                    is_train=False \
                )

        apex.amp.register_half_function(torch, 'einsum')
        self.model \
            = apex.amp.initialize(
                self.model, \
                opt_level="O1")
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_data(self, input_file, is_train):
        dpr_dataset = DPRPredDataset( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dpr_dataloader = DataLoader(
            dpr_dataset,
            shuffle=False,
            collate_fn=pred_data_collator,
            batch_size= \
                self.model_config.per_device_eval_batch_size * self.n_gpus
        )
        return (dpr_dataset, dpr_dataloader)
    
    def get_model(self):
        if "albert" in self.model_config.init_checkpoint:
            MODEL_CLASS = ALBERTForQsim
        elif "bert" in self.model_config.init_checkpoint:
            MODEL_CLASS = BERTForQsim
        else:
            MODEL_CLASS = DPRQuestionEncoder

        model = MODEL_CLASS.from_pretrained(
                self.model_config.init_checkpoint,
        )
        return model
    
    def pred_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", "attention_mask"
        ]
        em = []
        f1 = []
        scores = []
        self.model.eval()
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                bsize = batch["n_samples"]
                questions = batch["questions"]
                bm25_questions = batch["bm25_questions"]
                gt_answers = batch["gt_answer"]
                bm25_answers = batch["bm25_answers"]
                bm25_sim = batch["bm25_sim"]
                
                new_batch = {k: batch[k].to(self.device) for k in target_inputs}
                q_vecs = self.model(**new_batch) \
                    .pooler_output.detach().cpu().numpy()
                n_vecs, _ = q_vecs.shape
                queries = q_vecs[:bsize]
                
                chunk_size = batch["bm25_size"][0]
                n_chunks = (n_vecs - bsize) // chunk_size
                assert n_chunks * chunk_size == (n_vecs - bsize)
                bm25_vecs = np.split(q_vecs[bsize:], n_chunks)
                
                for question, q, cand, gt_answer, bm25_qs, bm25_answer, bm25_s in zip(questions, queries, bm25_vecs, gt_answers, bm25_questions, bm25_answers, bm25_sim):
                    sims = np.dot(cand, q)
                    pred_idx = np.argmax(sims)
                    
                    topk = np.argsort(-sims)[:1] #top K
                    pred_answer = []
                    for m_idx in topk:
                        pred_answer += bm25_answer[m_idx]
                    
                    f1.append(compute_f1_sets(set(gt_answer), set(pred_answer)))
                    em.append(compute_exact_sets(set(gt_answer), set(pred_answer)))
                    scores.append(sims[pred_idx])
        
        em = 100 * np.mean(em)
        f1 = 100 * np.mean(f1)
        print("EM: {:.04f} | F1: {:.04f}".format(em, f1))
        return (em, f1)

class DPRQsimPredInfSpeed:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.accelerator = Accelerator()
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.n_gpus = torch.cuda.device_count()
        print("N GPUs: {:d}".format(self.n_gpus))

        self.tokenizer =  \
            get_tokenizer( \
                self.model_config.init_checkpoint \
            )
        self.model = self.get_model()
        self.model = self.model.to(self.device) 

        self.eval_dataloader = None
        assert self.data_config.pred_file != None
        self.eval_dataset, \
            self.eval_dataloader = \
                self.get_data( \
                    self.data_config.pred_file, \
                    is_train=False \
                )

        apex.amp.register_half_function(torch, 'einsum')
        self.model \
            = apex.amp.initialize(
                self.model, \
                opt_level="O1")
        if self.n_gpus > 1:
            self.model \
                = torch.nn.DataParallel( \
                    self.model)
        self.model.eval()
    
    def get_data(self, input_file, is_train):
        dpr_dataset = DPRPredInfSpeedDataset( \
            input_file, \
            self.tokenizer, \
            self.data_config \
        )
        dpr_dataloader = DataLoader(
            dpr_dataset,
            shuffle=False,
            collate_fn=pred_data_inf_speed_collator,
            batch_size= \
                self.model_config.per_device_eval_batch_size * self.n_gpus
        )
        return (dpr_dataset, dpr_dataloader)
    
    def get_model(self):
        if "albert" in self.model_config.init_checkpoint:
            MODEL_CLASS = ALBERTForQsim
        elif "bert" in self.model_config.init_checkpoint:
            MODEL_CLASS = BERTForQsim
        else:
            MODEL_CLASS = DPRQuestionEncoder

        model = MODEL_CLASS.from_pretrained(
                self.model_config.init_checkpoint,
        )
        return model

    def pred_model(self):
        target_inputs = [ \
            "input_ids", "token_type_ids", "attention_mask"
        ]
        em = []
        f1 = []
        scores = []
        self.model.eval()
        t_model, t_cpu, t_index = 0, 0, 0
        for batch in tqdm(self.eval_dataloader, desc="Eval"):
            with torch.no_grad():
                bsize = batch["n_samples"]
                questions = batch["questions"]
                bm25_questions = batch["bm25_questions"]
                gt_answers = batch["gt_answer"]
                bm25_answers = batch["bm25_answers"]
                bm25_sim = batch["bm25_sim"]
                new_batch_q = {k: batch[k].to(self.device) for k in target_inputs}
                torch.cuda.synchronize()
                t_start = time.time()
                q_vecs = self.model(**new_batch_q).pooler_output
                torch.cuda.synchronize()
                t_end = time.time()
                queries = q_vecs.cpu().numpy()
                torch.cuda.synchronize()
                t_end_cpu = time.time()
                t_model += t_end - t_start
                t_cpu += t_end_cpu - t_start

                can_bsize = 512
                can_vecs = []

                can_batch_size_all=batch["bm25_input_ids"].shape[0]
                
                for batch_start in range(0, can_batch_size_all, can_bsize):
                    new_batch_can = {k: batch["bm25_{}".format(k)][batch_start: batch_start+can_bsize].to(self.device) for k in target_inputs}
                    can_vec = self.model(**new_batch_can).pooler_output.detach().cpu().numpy()
                    can_vecs.append(can_vec)

                can_vecs = np.concatenate(can_vecs, axis=0)
                
                chunk_size = batch["bm25_size"][0]
                n_chunks = can_vecs.shape[0] // chunk_size
               
                
                assert n_chunks * chunk_size == can_vecs.shape[0]
                bm25_vecs = np.split(can_vecs, n_chunks)
                
                for question, q, cand, gt_answer, bm25_qs, bm25_answer, bm25_s in zip(questions, queries, bm25_vecs, gt_answers, bm25_questions, bm25_answers, bm25_sim):
                    t_index_start = time.time()
                    sims = np.dot(cand, q)
                    pred_idx = np.argmax(sims)
                    
                    topk = np.argsort(-sims)[:1] #top K
                    pred_answer = []
                    for m_idx in topk:
                        pred_answer += bm25_answer[m_idx]
                    t_index_end = time.time()
                    t_index += t_index_end - t_index_start

                    f1.append(compute_f1_sets(set(gt_answer), set(pred_answer)))
                    em.append(compute_exact_sets(set(gt_answer), set(pred_answer)))
                    scores.append(sims[pred_idx])
        
        print(f"TIME for model: {t_model} secs / 3610 questions")
        print(f"TIME for model + cpu: {t_cpu} secs / 3610 questions")
        print(f"TIME for indexing: {t_index} secs / 3610 questions")

        em = 100 * np.mean(em)
        f1 = 100 * np.mean(f1)
        print("EM: {:.04f} | F1: {:.04f}".format(em, f1))
        
        return (em, f1)
