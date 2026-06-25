from time import time
import os
import shutil
from pathlib import Path
from typing import Iterable, Callable

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from ..data.dataloader import DataLoader
from ..data.query import Query
from ..logic.spl_module import SoftProofModule, DebugSoftProofModule
from .logger import PrintLogger, WandbLogger
from .loss import nll_loss, get_optimizer
from .metrics import get_metrics, aggregate_metrics
from . import set_seed, ConfigDict
from deepsoftlog.data import expression_to_prolog, query_to_prolog

##Import F
import torch.nn.functional as F
import re
from collections import defaultdict
from time import time

def ddp_setup(rank, world_size):
    print(f"Starting worker {rank + 1}/{world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    set_seed(1532 + rank)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _trainp(rank, world_size, trainer, cfg):
    ddp_setup(rank, world_size)
    trainer.program.store = DDP(trainer.program.store, find_unused_parameters=True)
    trainer._train(cfg, master=rank == 0)
    dist.destroy_process_group()


# Add before your optimization step
def get_embedding_debug_info(store, term1, term2):
    result = {}
    
    # Check if embeddings exist before accessing them
    if hasattr(store.constant_embeddings, term1) and hasattr(store.constant_embeddings, term2):
        emb1 = store.constant_embeddings[term1].clone().detach()
        emb2 = store.constant_embeddings[term2].clone().detach()
        
        # Store the initial embeddings
        result['emb1_before'] = emb1
        result['emb2_before'] = emb2
        result['sim_before'] = F.cosine_similarity(emb1, emb2, dim=0).item()
        
    return result

class Trainer:
    def __init__(
            self,
            program: SoftProofModule,
            load_train_dataset: Callable[[dict], DataLoader],
            criterion,
            optimizer: Optimizer,
            logger=PrintLogger(),
            **search_args
    ):
        self.program = program
        self.program.mask_query = True
        self.logger = logger
        self.load_train_dataset = load_train_dataset
        self.train_dataset = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = None
        self.grad_clip = None
        self.search_args = search_args
        self.current_epoch = 0

    def check_soft_unifications_in_proof(self, proof_result):
        """Extract soft unification statistics from a proof result"""
        soft_unifs = {}
        
        # Check if there are any soft unifications in the proof
        if hasattr(proof_result, 'facts'):
            # The structure might vary - adjust based on your actual proof result structure
            facts_dict = proof_result.facts
            
            # Look for soft unification facts which typically have the form k(term1,term2)
            for fact_key in facts_dict:
                if isinstance(fact_key, str) and 'k(' in fact_key:
                    terms_match = re.search(r'k\(([^,]+),([^)]+)\)', fact_key)
                    if terms_match:
                        term1, term2 = terms_match.groups()
                        key = f"{term1}_{term2}"
                        prob = facts_dict[fact_key]
                        soft_unifs[key] = prob
        
        return soft_unifs
    
    def _train(self, cfg: dict, master=True, do_eval=True):
        nb_epochs = cfg['nb_epochs']
        self.grad_clip = cfg['grad_clip']
        self.program.store.to(cfg['device'])
        self.program.store.train()
        self.train_dataset = self.load_train_dataset(cfg)
        self.scheduler = CosineAnnealingLR(self.optimizer, nb_epochs + 1)
        for epoch in range(nb_epochs):
            last_lr = self.scheduler.get_last_lr()[0]
            print(f"### EPOCH {epoch} (lr={last_lr:.2g}) ###")
            self.train_epoch(verbose=cfg['verbose'] and master)
            self.scheduler.step()
            if master:
                self.save(cfg)
            if do_eval and master and hasattr(self, 'val_dataloader'):
                self.eval(self.val_dataloader, name='val')
            print(f"Program algebra:\n{self.program.algebra._sdd_algebra.all_facts._val_to_ix}\n") if cfg['verbose'] else print(self.program)

    def train(self, cfg: dict, nb_workers: int = 1):
        if nb_workers == 1:
            return self._train(cfg, True)
        self.program.algebra = None
        self.train_dataset = None
        mp.spawn(_trainp,
                 args=(nb_workers, self, cfg),
                 nprocs=nb_workers,
                 join=True)

    def train_profile(self, *args, **kwargs):
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        self.train(*args, **kwargs)
        profiler.stop()
        profiler.open_in_browser()

        
    #         loss, diff, proof_steps, nb_proofs = self.get_loss(queries)
    #                 'grad_norm': grad_norm,
    #                 'loss': loss,
    #                 'diff': diff,
    #                 "step_time": time() - current_time,
    #                 "proof_steps": proof_steps,
    #                 "nb_proofs": nb_proofs,
    #             })
    
    # Add this function to your class - outside train_epoch
    def debug_soft_unifications_in_proof(self, proof_result, query):
        """Extract detailed soft unification statistics from a proof result"""
        
        # Track all soft unifications across all proofs
        all_soft_unifs = {}
        
        # Print proof result structure
        
        # For dictionary results (which is what SoftProofModule.query returns)
        if isinstance(proof_result, dict):
            
            # Look for keys with 'k(' in them - these are soft unification facts
            for key in proof_result.keys():
                key_str = str(key)
                if 'k(' in key_str:
                    try:
                        # Parse out term1 and term2 from k(term1,term2)
                        terms_match = re.search(r'k\(([^,]+),([^)]+)\)', key_str)
                        if terms_match:
                            term1, term2 = terms_match.groups()
                            unif_key = f"{term1}_{term2}"
                            # Get the probability value
                            prob_value = proof_result[key]
                            if isinstance(prob_value, torch.Tensor):
                                prob = prob_value.item()
                            else:
                                prob = float(prob_value)
                            all_soft_unifs[unif_key] = prob
                    except Exception as e:
                        print(f"Error processing key {key}: {e}")
        
        # Check embedding parameters and their gradients
                    
        #             # Show gradient info if available
                    
        #             # Fix embeddings that don't require gradients
        
        # # Print summary of findings
        
        return all_soft_unifs

    # Add these helper functions - outside your class
    def extract_soft_unifs_from_proof(proof, result_dict):
        """Helper to extract soft unifications from a proof structure"""
        if hasattr(proof, 'facts'):
            facts = proof.facts
            try:
                # Different ways facts might be structured
                if hasattr(facts, 'positive_soft_facts'):
                    for fact in facts.positive_soft_facts:
                        if 'k(' in str(fact):
                            terms_match = re.search(r'k\(([^,]+),([^)]+)\)', str(fact))
                            if terms_match:
                                term1, term2 = terms_match.groups()
                                key = f"{term1}_{term2}"
                                prob = float(str(fact).split(':')[0])
                                result_dict[key] = prob
                elif hasattr(facts, 'items'):
                    for key, value in facts.items():
                        if 'k(' in str(key):
                            terms_match = re.search(r'k\(([^,]+),([^)]+)\)', str(key))
                            if terms_match:
                                term1, term2 = terms_match.groups()
                                result_key = f"{term1}_{term2}"
                                result_dict[result_key] = value
            except Exception as e:
                print(f"  Error extracting soft unifications: {e}")

    def check_embedding_parameters(store, terms):
        """Check if embeddings are properly registered parameters"""
        for term in terms:
            try:
                if term in store.constant_embeddings:
                    emb = store.constant_embeddings[term]
                    print(f"  {term}:")
                    if emb.grad is not None:
                        print(f"    Grad mean: {emb.grad.mean().item()}")
                else:
                    print(f"  {term}: Not found in constant_embeddings")
            except Exception as e:
                print(f"  Error checking {term}: {e}")

    def check_gradient_flow(self, program):
        """Check which gradients are flowing and their magnitudes"""
        total_params = 0
        params_without_grad = 0
        
        
        # Check embeddings in the store
        if hasattr(program, 'store') and hasattr(program.store, 'constant_embeddings'):
            for name, param in program.store.constant_embeddings.items():
                total_params += 1
                if not hasattr(param, 'grad') or param.grad is None:
                    params_without_grad += 1
                else:
                    grad_norm = torch.norm(param.grad).item()
        
        # Check all parameters using the parameters() method
        try:
            for param in program.parameters():
                total_params += 1
                if param.grad is None:
                    params_without_grad += 1
                else:
                    grad_norm = torch.norm(param.grad).item()
        except Exception as e:
            print(f"Error checking parameters: {e}")
        

    def get_embedding_debug_info(store, term1, term2):
        result = {}
        
        # Check if embeddings exist as dictionary keys, not attributes
        if term1 in store.constant_embeddings and term2 in store.constant_embeddings:
            emb1 = store.constant_embeddings[term1].clone().detach()
            emb2 = store.constant_embeddings[term2].clone().detach()
            
            # Store the initial embeddings
            result['emb1_before'] = emb1
            result['emb2_before'] = emb2
            result['sim_before'] = F.cosine_similarity(emb1, emb2, dim=0).item()
        
        return result
    
    def monitor_embeddings(self, epoch, batch_idx):
        """Log embedding relationships and gradients over training"""
        store = self.program.store
        
        results = {
            'epoch': epoch,
            'batch': batch_idx,
            'similarities': {},
            'gradients': {}
        }
        
        # Track key similarities
        term_pairs = [
            ('dog', 'animal'),
            ('cat', 'animal'),
            ('man', 'person')
        ]
        
        for term1, term2 in term_pairs:
            if term1 in store.constant_embeddings and term2 in store.constant_embeddings:
                emb1 = store.constant_embeddings[term1]
                emb2 = store.constant_embeddings[term2]
                sim = F.cosine_similarity(emb1, emb2, dim=0).item()
                results['similarities'][f"{term1}_{term2}"] = sim
        
        # Track gradient norms
        key_terms = ['dog', 'cat', 'animal', 'person', 'man']
        for term in key_terms:
            if term in store.constant_embeddings:
                emb = store.constant_embeddings[term]
                if emb.grad is not None:
                    grad_norm = torch.norm(emb.grad).item()
                    results['gradients'][term] = grad_norm
                else:
                    results['gradients'][term] = 0.0
        
        # Log to file
        with open("embedding_monitor.csv", "a") as f:
            if epoch == 0 and batch_idx == 0:
                # Write header on first call
                f.write("epoch,batch,")
                for pair in term_pairs:
                    f.write(f"sim_{pair[0]}_{pair[1]},")
                for term in key_terms:
                    f.write(f"grad_{term}" + ("," if term != key_terms[-1] else "\n"))
            
            # Write data row
            f.write(f"{epoch},{batch_idx},")
            for pair in term_pairs:
                pair_key = f"{pair[0]}_{pair[1]}"
                f.write(f"{results['similarities'].get(pair_key, 0.0)},")
            for term in key_terms:
                f.write(f"{results['gradients'].get(term, 0.0)}" + ("," if term != key_terms[-1] else "\n"))
        
        return results
    
    def debug_instance_evaluations(self):
        """Directly examine each training instance's evaluation"""
        
        # Get the dataset (assuming train_dataset is iterable)
        for batch_idx, batch in enumerate(self.train_dataset):
            print(f"\nBatch {batch_idx}:")
            
            # Process each instance in the batch
            for instance_idx, instance in enumerate(batch):
                print(f"\n  Instance {instance_idx}:")
                
                # Extract the query directly
                if hasattr(instance, 'query'):
                    query = instance.query
                    
                    # If there's a target probability
                    if hasattr(instance.query, 'p'):
                        print(f"    Target probability: {instance.query.p}")
                    
                    # Execute the query directly
                    try:
                        result = self.program.query(query.query)
                        print(f"    Result: {result}")
                        
                        # Check for soft unifications
                        if isinstance(result, dict):
                            print(result)
                            for key, value in result.items():
                                if 'k(' in str(key):
                                    print(f"      {key}: {value}")
                        
                        # Try to calculate diff directly
                        if hasattr(instance, 'error_with'):
                            try:
                                diff = instance.error_with(result)
                                print(f"    Diff: {diff}")
                            except Exception as e:
                                print(f"    Error calculating diff: {e}")
                        else:
                            print("    No error_with attribute found")
                    except Exception as e:
                        print(f"    Error executing query: {e}")
                else:
                    print(f"    No query attribute found. Instance attributes: {dir(instance)}")
        


    #     """Training epoch with proper handling of DatasetInstance objects"""
    #     # Clear the soft unification cache at the start of the epoch
        
    #     # Initialize tracking
        
            
    #         # Track embedding similarities before optimization
                
    #                     similarities[f'{term1}_{term2}'] = sim
            
    #         # Debug: print instance and query structure
                    
    #                 # Access the query properly
                        
    #                     # Convert to prolog query if needed
                            
                        
    #                     # Try executing the query
                            
    #                         # Look for soft unifications in the result
            
    #         # Normal loss calculation and optimization
    #         loss, diff, proof_steps, nb_proofs = self.get_loss(instances)
    #         # Check gradients before optimizer step
                
            
    #         # Perform optimization step
                
    #         # Check similarities after optimization
                
    #                 term1, term2 = pair.split('_')
            
    #         # Standard logging
    #                 'grad_norm': grad_norm,
    #                 'loss': loss,
    #                 'diff': diff,
    #                 "step_time": time() - current_time,
    #                 "proof_steps": proof_steps,
    #                 "nb_proofs": nb_proofs,
    #             })
        
    #     # At the end of the epoch
        
        
    #     # Increment epoch counter if you're tracking it
    
    
    def train_epoch(self, verbose: bool):
        """Training epoch with CSV logging of metrics"""
        # Clear the soft unification cache at the start of the epoch
        if hasattr(self.program, 'soft_unification_cache'):
            self.program.soft_unification_cache = {}
        
        # Initialize or update epoch counter
        if not hasattr(self, 'current_epoch'):
            self.current_epoch = 0
        
        # Setup CSV file if it doesn't exist
        csv_file = 'training_metrics_final.csv'
        create_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
        
        with open(csv_file, 'a') as f:
            if create_header:
                f.write('epoch,batch,loss,diff,proof_steps,nb_proofs,query,target\n')
        
        for batch_idx, instances in enumerate(tqdm(self.train_dataset, leave=False, smoothing=0, disable=not verbose)):
            current_time = time()
            

            
            # Normal loss calculation and optimization
            loss, diff, proof_steps, nb_proofs = self.get_loss(instances)
            
            # Print metrics
            print(f"Epoch {self.current_epoch}, Batch {batch_idx}")
            print(f"Loss: {loss}, Diff: {diff}, Proof steps: {proof_steps}, Number of proofs: {nb_proofs}")
            
            # Get query and target info
            query_str = str(instances[0].query) if hasattr(instances[0], 'query') else 'unknown'
            target_str = str(instances[0].target) if hasattr(instances[0], 'target') else 'unknown'
            
            # Log to CSV
            with open(csv_file, 'a') as f:
                # Replace commas and newlines in strings to avoid CSV issues
                safe_query = query_str.replace(',', ';').replace('\n', ' ')
                safe_target = target_str.replace(',', ';').replace('\n', ' ')
                
                f.write(f"{self.current_epoch},{batch_idx},{loss},{diff},{proof_steps},{nb_proofs},")
                f.write(f"\"{safe_query}\",\"{safe_target}\"\n")
            
            # Perform optimization step
            grad_norm = 0.
            if loss is not None:
                grad_norm = self.step_optimizer()
            
            # Standard logging
            if verbose:
                self.logger.log({
                    'grad_norm': grad_norm,
                    'loss': loss,
                    'diff': diff,
                    "step_time": time() - current_time,
                    "proof_steps": proof_steps,
                    "nb_proofs": nb_proofs,
                })
        
        # Increment epoch counter
        self.current_epoch += 1
        
        if verbose:
            self.logger.print()
        print("EPOCH END")
    
        # ADD THIS LINE:
        self.check_embedding_changes(self.current_epoch - 1)

    #         metrics += new_metrics
    
    def eval(self, dataloader: DataLoader, name='test'):
        print("EVALUATION STARTING")
        self.program.store.eval()
        metrics = []
        print(f"DataLoader: {dataloader}")
        for instance in dataloader.dataset.instances:
            if not isinstance(instance.query, Query):
                instance.query = query_to_prolog(instance.query)

        for queries in tqdm(dataloader, leave=False, smoothing=0):

            if not isinstance(queries, Query): 
                queries = [q.query for q in queries]

            results = zip(queries, self._eval_queries(queries))
            print(f"Results: {results}")

            # Ensure we only access .query if it's still a DatasetInstance
            new_metrics = [get_metrics(query, result, queries) for query, result in results]

            metrics += new_metrics

        self.logger.log_eval(aggregate_metrics(metrics), name=name)

    def _query(self, queries: Iterable[Query]):
        for query in queries:
            print(f"Query: {query.query}")
            result, proof_steps, nb_proofs = self.program(
                query.query, **self.search_args
            )
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            yield result, proof_steps, nb_proofs

        
    
    def _query_result(self, queries: Iterable[Query]):
        for query in queries:
            print("_query_result called, type:")
            print(query)
            yield self.program.query(query.query, **self.search_args)

    def _eval_queries(self, queries: Iterable[Query]):
        with torch.no_grad():
            for query, results in zip(queries, self._query_result(queries)):
                if len(results) > 0:
                    results = {k: v.exp().item() for k, v in results.items() if v != 0.}
                    yield results
                else:
                    print(f"WARNING: empty result for {query}")
                    yield {}

    def get_loss(self, queries: Iterable[Query]) -> tuple[float, float, float, float]:
        print(f"Queries: {queries[:5]}")
        results, proof_steps, nb_proofs = tuple(zip(*self._query(queries)))
        print(f"Results: {results}")
        print(f"Proof steps: {proof_steps}")
        print(f"Number of proofs: {nb_proofs}")
        losses = [self.criterion(result, query.p) for result, query in zip(results, queries)]
        print(f"Losses: {losses}")
        loss = torch.stack(losses).mean()
        errors = [query.error_with(result) for result, query in zip(results, queries)]
        if loss.requires_grad:
            loss.backward()
        proof_steps, nb_proofs = float(np.mean(proof_steps)), float(np.mean(nb_proofs))
        return float(loss), float(np.mean(errors)), proof_steps, nb_proofs

    
    #             torch.nn.utils.clip_grad_norm_(self.program.parameters(), max_norm=self.grad_clip)
    
    def step_optimizer(self):
        # Print gradient norms for different parts of the model
        
        # Check embedding gradients
        empty_grads = 0
        total_params = 0
        for name, param in self.program.store.named_parameters():
            total_params += 1
            if param.grad is None:
                empty_grads += 1
            else:
                grad_norm = param.grad.norm().item()
        
        print(f"{empty_grads} out of {total_params} parameters have no gradients")
        
        # Continue with the existing code
        with torch.no_grad():
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.program.parameters(), max_norm=self.grad_clip)
            grad_norm = self.program.grad_norm()
        # In your trainer, right before optimizer.step():
        print(f"About to step optimizer with {sum(len(g['params']) for g in self.optimizer.param_groups)} parameters")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.get_store().clear_cache()
        return float(grad_norm)


    def save(self, config: ConfigDict):
        save_folder = f"results/{config['name']}"
        save_folder = Path(save_folder)
        if save_folder.exists():
            shutil.rmtree(save_folder, ignore_errors=True)
        save_folder.mkdir(parents=True)

        config.save(save_folder / "config.yaml")
        torch.save(self.get_store().state_dict(), save_folder / "store.pt")

    def get_store(self):
        return self.program.get_store()
    
    #     """Update optimizer if new parameters were added"""
        
    
    def update_optimizer_if_needed(self):
        """Update optimizer if new parameters were added"""
        current_param_count = sum(len(group['params']) for group in self.optimizer.param_groups)
        expected_constant_params = len(self.program.store.constant_embeddings)
        expected_functor_params = sum(len(list(model.parameters())) for model in self.program.store.functor_embeddings.values())
        expected_total = expected_constant_params + expected_functor_params
        
        if current_param_count < expected_total:
            
            # Extract config from existing optimizer
            config = {
                'optimizer': 'AdamW',
                'embedding_learning_rate': self.optimizer.param_groups[0]['lr'],
                'functor_learning_rate': self.optimizer.param_groups[1]['lr'],
                'functor_weight_decay': self.optimizer.param_groups[1].get('weight_decay', 0.)
            }
            
            from deepsoftlog.training.loss import get_optimizer
            self.optimizer = get_optimizer(self.program.get_store(), config)
    def save_pretrained_model(self, save_path):
        """Save all components needed to reload pretrained model"""
        
        checkpoint = {
            # Main learned parameters
            'embedding_store_state_dict': self.program.store.state_dict(),
            
            # Model configuration
            'embedding_dimensions': self.program.store.ndim,
            'embedding_metric': self.program.embedding_metric,
            'semantics': self.program.semantics,
            
            'vocabulary_constants': list(self.program.store.vocabulary.get_constants()),
            'vocabulary_functors': [str(sig) for sig in self.program.store.vocabulary.get_functors()],
            
            # Training info (optional but useful)
            'training_epochs': self.current_epoch,
            'final_loss': getattr(self, 'final_loss', None),
        }
        
        torch.save(checkpoint, save_path)
        print(f"Saved pretrained model to {save_path}")
        
    #     """Load pretrained model with debugging"""
        
        
    #     # Parse the base program
    #         initial_program_path,
    #     )
        
    #     # Recreate vocabulary
    #         vocab.add_constant(const)
    #             vocab.add_functor(eval(functor_str))
        
        
    #     # Create store
        
        
        
    #     # Check a few embedding values BEFORE loading
        
    #     # Load state dict
    #     missing_keys, unexpected_keys = new_store.load_state_dict(checkpoint['embedding_store_state_dict'], strict=False)
        
    #     # Check the same embedding AFTER loading
        
        
    #     pretrained_program.store = new_store
    
    
    def load_pretrained_model(self, checkpoint_path, initial_program_path='initial_program.pl'):
        """Load pretrained model with debugging"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"🔍 Step 1: Loaded checkpoint")
        
        # Parse the base program
        from deepsoftlog.parser.parser import parse_file
        pretrained_program = parse_file(
            initial_program_path,
            embedding_metric=checkpoint['embedding_metric'],
            semantics=checkpoint['semantics'],
        )
        
        # Create new store with loaded embeddings
        from deepsoftlog.parser.vocabulary import Vocabulary
        vocab = Vocabulary()
        for const in checkpoint['vocabulary_constants']:
            vocab.add_constant(const)
        for functor_str in checkpoint['vocabulary_functors']:
            try:
                vocab.add_functor(eval(functor_str))
            except:
                pass
        
        from deepsoftlog.embeddings.initialize_vector import Initializer
        from deepsoftlog.embeddings.nn_models import EmbeddingFunctor
        from deepsoftlog.embeddings.embedding_store import EmbeddingStore
        
        initializer = Initializer(EmbeddingFunctor, 'uniform', checkpoint['embedding_dimensions'])
        new_store = EmbeddingStore(checkpoint['embedding_dimensions'], initializer, vocab)
        
        
        # Load state dict
        missing_keys, unexpected_keys = new_store.load_state_dict(checkpoint['embedding_store_state_dict'], strict=False)
        
        # Check the store assignment
        
        pretrained_program.store = new_store
        
        print(f"🔍 Step 6: Are they the same object? {pretrained_program.store is new_store}")
        
        # Test get_store() too
        get_store_result = pretrained_program.get_store()
        
        return pretrained_program   



def create_trainer(program, load_train_dataset, cfg):
    trainer_args = {
        "program": program,
        "criterion": nll_loss,
        "load_train_dataset": load_train_dataset,
        "optimizer": get_optimizer(program.get_store(), cfg),
        "logger": WandbLogger(cfg),
        "max_proofs": cfg.get("max_proofs", None),
        "max_depth": cfg.get("max_depth", None),
        "max_branching": cfg.get("max_branching", None),
    }
    return Trainer(**trainer_args)
