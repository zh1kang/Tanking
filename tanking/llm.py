"""vLLM wrapper: sample thinking traces, then read the action distribution after a forced `Action:`.

A trace's observable thinking time is its number of generated thinking tokens (the closing </think> excluded).
Traces that hit the cap, or are cut to a budget, are closed with an early-stop sentence before the answer.
"""

from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from tanking.prompts import ANSWER_PREFIX, EARLY_STOP

ACTIONS = ("check", "bet", "call", "raise", "fold")
THINK_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)


@dataclass
class Answer:
    probs: np.ndarray
    legal_mass: float


class Engine:
    def __init__(self, model: str, max_model_len: int = 4096, seed: int = 0, gpu_memory_utilization: float = 0.85):
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_logprobs=20,
        )
        self.tok = self.llm.get_tokenizer()
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.action_id = {}
        for w in ACTIONS:
            ids = self.tok.encode(" " + w, add_special_tokens=False)
            assert len(ids) == 1, f"' {w}' must be one token, got {ids}"
            self.action_id[w] = ids[0]
        self.close_ids = self.tok.encode("</think>\n\n" + ANSWER_PREFIX, add_special_tokens=False)
        self.early_ids = self.tok.encode(EARLY_STOP, add_special_tokens=False)
        self.answer_ids = self.tok.encode(ANSWER_PREFIX, add_special_tokens=False)

    def prompt(self, message: str, think: bool) -> list[int]:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
        return self.tok.encode(text, add_special_tokens=False)

    def think(
        self, prompts: list[list[int]], n: int, max_think: int | list[int], seed: int = 0, use_tqdm: bool = True
    ) -> list[list[tuple[list[int], bool]]]:
        """n traces per prompt as (thinking token ids, finished before the cap); the cap may differ per prompt."""
        caps = max_think if isinstance(max_think, list) else [max_think] * len(prompts)
        sp = [
            SamplingParams(n=n, max_tokens=c, stop_token_ids=[self.think_end], seed=seed, **THINK_SAMPLING)
            for c in caps
        ]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=use_tqdm)
        result = []
        for o in outs:
            traces = []
            for c in o.outputs:
                ids = list(c.token_ids)
                done = c.finish_reason == "stop"
                if ids and ids[-1] == self.think_end:
                    ids = ids[:-1]
                traces.append((ids, done))
            result.append(traces)
        return result

    def answer_prefix(self, prompt: list[int], thought: list[int], closed: bool) -> list[int]:
        return prompt + thought + (self.close_ids if closed else self.early_ids)

    def answers(self, prefixes: list[list[int]], words: list[list[str]]) -> list[Answer]:
        """Distribution over each prefix's legal action words, renormalized from the next-token top-20 logprobs."""
        sp = SamplingParams(max_tokens=1, logprobs=20, temperature=1.0)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes], sp, use_tqdm=False)
        res = []
        for o, ws in zip(outs, words):
            lp = o.outputs[0].logprobs[0]
            x = np.array([lp[self.action_id[w]].logprob if self.action_id[w] in lp else -np.inf for w in ws])
            m = np.exp(x)
            total = m.sum()
            res.append(Answer(m / total if total > 0 else np.full(len(ws), 1 / len(ws)), float(total)))
        return res
