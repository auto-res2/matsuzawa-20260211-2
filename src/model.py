"""Model loading and inference utilities for CISC experiment."""

import torch
from typing import List, Optional, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.preprocess import extract_final_number


class InferenceEngine:
    """Inference engine for CoT-based methods."""
    
    def __init__(
        self,
        model_name: str,
        cache_dir: str = ".cache",
        device: str = "auto",
        dtype: str = "auto",
    ):
        """
        Initialize inference engine.
        
        Args:
            model_name: Hugging Face model name
            cache_dir: Directory to cache model weights
            device: Device to load model on ("auto", "cuda", "cpu")
            dtype: Data type ("auto", "bf16", "fp16", "fp32")
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Determine dtype
        if dtype == "auto":
            if self.device == "cuda":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
        elif dtype == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated text (excluding prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else 1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        return full_text[len(prompt):]
    
    @torch.no_grad()
    def sample_cot_answers(
        self,
        question: str,
        k: int,
        temperature: float,
        max_new_tokens: int,
        cot_trigger: str = "Let's think step by step",
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Sample k CoT completions and extract final answers.
        
        Args:
            question: Question to solve
            k: Number of samples
            temperature: Sampling temperature
            max_new_tokens: Max tokens per sample
            cot_trigger: CoT trigger phrase
            
        Returns:
            List of (completion, answer) tuples
        """
        prompt = (
            "You are a careful reasoner. Solve step by step, then end with 'Final answer: <number>'.\n"
            f"Q: {question}\n"
            f"A: {cot_trigger}.\n"
        )
        
        samples = []
        for _ in range(k):
            completion = self.generate_text(prompt, temperature, max_new_tokens)
            answer = extract_final_number(completion)
            samples.append((completion, answer))
        
        return samples
    
    @torch.no_grad()
    def get_yes_no_probability(self, prompt: str) -> float:
        """
        Get probability of "Yes" vs "No" from next token distribution.
        
        Args:
            prompt: Prompt for yes/no question
            
        Returns:
            P(Yes) in [0, 1]
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1]
        
        # Get token IDs for Yes and No
        yes_ids = self._get_first_token_ids("Yes")
        no_ids = self._get_first_token_ids("No")
        
        # Calculate probabilities
        candidate_ids = yes_ids + no_ids
        probs = torch.softmax(logits[candidate_ids], dim=-1)
        
        yes_mass = sum(probs[i].item() for i in range(len(yes_ids)))
        no_mass = sum(probs[i].item() for i in range(len(yes_ids), len(candidate_ids)))
        
        return yes_mass / (yes_mass + no_mass + 1e-12)
    
    def _get_first_token_ids(self, text: str) -> List[int]:
        """Get token IDs for first token of text (with and without leading space)."""
        ids = []
        for variant in [text, " " + text]:
            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.append(encoded[0])
        return list(dict.fromkeys(ids))  # Remove duplicates
    
    @torch.no_grad()
    def blind_judge(self, question: str, answer: str) -> float:
        """
        Blind plausibility judge: P(answer is correct | question, answer).
        
        Does not see the reasoning/rationale.
        
        Args:
            question: Question text
            answer: Proposed answer
            
        Returns:
            Probability that answer is correct
        """
        prompt = (
            "You are verifying an answer. Reply with exactly one word: Yes or No.\n"
            f"Q: {question}\n"
            f"Proposed final answer: {answer}\n"
            "Is the proposed final answer correct?"
        )
        
        return self.get_yes_no_probability(prompt)
    
    @torch.no_grad()
    def paraphrase_question(
        self,
        question: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """
        Generate semantics-preserving paraphrase of question.
        
        Args:
            question: Original question
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            
        Returns:
            Paraphrased question
        """
        prompt = (
            "Rewrite the following math word problem using different wording and sentence order, "
            "but keep the exact same meaning, quantities, and constraints. Do NOT solve it. "
            "Output ONLY the rewritten question.\n"
            f"Original question: {question}\n"
            "Rewritten question:"
        )
        
        text = self.generate_text(prompt, temperature, max_new_tokens)
        
        # Keep only first line to avoid model adding commentary
        return text.strip().split("\n")[0].strip()
    
    @torch.no_grad()
    def solve_answer_only(
        self,
        question: str,
        temperature: float,
        max_new_tokens: int,
    ) -> Optional[str]:
        """
        Solve question concisely without full CoT.
        
        Args:
            question: Question to solve
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            
        Returns:
            Extracted final answer
        """
        prompt = (
            "Solve the problem. Be concise. End with exactly: 'Final answer: <number>'.\n"
            f"Q: {question}\n"
            "A:"
        )
        
        completion = self.generate_text(prompt, temperature, max_new_tokens)
        return extract_final_number(completion)


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """
    Select most frequent answer (majority vote).
    
    Args:
        answers: List of answers
        
    Returns:
        Most frequent answer, or None if all None
    """
    counts = defaultdict(int)
    for ans in answers:
        if ans is not None:
            counts[ans] += 1
    
    if not counts:
        return None
    
    return max(counts.items(), key=lambda x: x[1])[0]


def self_consistency_select(
    samples: List[Tuple[str, Optional[str]]],
) -> Optional[str]:
    """
    Self-consistency: majority vote over sampled answers.
    
    Args:
        samples: List of (completion, answer) tuples
        
    Returns:
        Selected answer
    """
    answers = [ans for _, ans in samples]
    return majority_vote(answers)


def cisc_select(
    engine: InferenceEngine,
    question: str,
    samples: List[Tuple[str, Optional[str]]],
    t_paraphrase: int,
    temp_paraphrase: float,
    max_tokens_paraphrase: int,
    temp_solve: float,
    max_tokens_solve: int,
    alpha: float = 1.0,
    eps: float = 0.2,
) -> Optional[str]:
    """
    CISC selection: counterfactual invariance self-consistency.
    
    Args:
        engine: Inference engine
        question: Original question
        samples: List of (completion, answer) tuples from proposal stage
        t_paraphrase: Number of paraphrases
        temp_paraphrase: Temperature for paraphrasing
        max_tokens_paraphrase: Max tokens for paraphrases
        temp_solve: Temperature for re-solving
        max_tokens_solve: Max tokens for re-solving
        alpha: Blind judge weight exponent
        eps: Invariance floor
        
    Returns:
        Selected answer
    """
    # Build unique answers and counts
    counts = defaultdict(int)
    for _, ans in samples:
        if ans is not None:
            counts[ans] += 1
    
    if not counts:
        return None
    
    # Generate paraphrases (shared across all candidates)
    paraphrases = [
        engine.paraphrase_question(question, temp_paraphrase, max_tokens_paraphrase)
        for _ in range(t_paraphrase)
    ]
    
    # Solve each paraphrase (shared)
    paraphrase_solutions = [
        engine.solve_answer_only(pq, temp_solve, max_tokens_solve)
        for pq in paraphrases
    ]
    
    # Score each unique candidate
    scores = {}
    for ans, count in counts.items():
        # Blind plausibility
        p_yes = engine.blind_judge(question, ans)
        
        # Invariance score: fraction of paraphrase solutions matching this answer
        num_match = sum(int(sol == ans) for sol in paraphrase_solutions if sol is not None)
        inv_score = num_match / max(1, len(paraphrase_solutions))
        
        # Combined score
        score = (
            torch.log(torch.tensor(1.0 + count)).item()
            * (p_yes ** alpha)
            * (eps + (1.0 - eps) * inv_score)
        )
        
        scores[ans] = score
    
    return max(scores.items(), key=lambda x: x[1])[0]
