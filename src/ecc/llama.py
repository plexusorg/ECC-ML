"""`python -m ecc.llama SETTINGS`: the classifier's margins for whole conversation windows, through llama.cpp's C API.

`uv run eval` starts it as its own process: the llama.cpp build's OpenMP runtime cannot share a process with
PyTorch's, which transformers loads. SETTINGS (JSON): build (the folder of the [eval] server build, whose llama.dll is
loaded), gguf, seqs, seq_tokens, ubatch, threads. Each stdin line is one conversation, its messages in order as
[[key, prompt], ...]; each stdout line is [key, margin]. llama.cpp logs to stderr.

A message's prompt is its conversation window up to its own line (chat.fitted_prompt). While each prompt's tokens extend
the previous one's, the messages share one window: one forward pass over the last prompt has every earlier message's
score at the position of its last token, as causal attention sees only the tokens before it. A decode holds up to
`seqs` windows, each in its own sequence of `seq_tokens` tokens, the deployment's slot size.
"""
import ctypes as C
import json
import os
import sys

import numpy as np

# llama-server's n_probs, as the model host requests it: an answer token outside the top TOP_LOGPROBS is scored at the
# lowest of them
TOP_LOGPROBS = 20


class ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p), ("n_gpu_layers", C.c_int32),
                ("split_mode", C.c_int), ("load_mode", C.c_int), ("lazy_mode", C.c_int), ("main_gpu", C.c_int32),
                ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p),
                ("progress_callback_user_data", C.c_void_p), ("kv_overrides", C.c_void_p),
                ("vocab_only", C.c_bool), ("check_tensors", C.c_bool), ("use_extra_bufts", C.c_bool),
                ("no_host", C.c_bool), ("no_alloc", C.c_bool), ("load_mtp", C.c_bool)]


class ContextParams(C.Structure):
    _fields_ = [("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32), ("n_seq_max", C.c_uint32),
                ("n_rs_seq", C.c_uint32), ("n_outputs_max", C.c_uint32), ("n_outputs_max_per_seq", C.c_uint32),
                ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32), ("ctx_type", C.c_int),
                ("rope_scaling_type", C.c_int), ("pooling_type", C.c_int), ("attention_type", C.c_int),
                ("flash_attn_type", C.c_int), ("rope_freq_base", C.c_float), ("rope_freq_scale", C.c_float),
                ("yarn_ext_factor", C.c_float), ("yarn_attn_factor", C.c_float), ("yarn_beta_fast", C.c_float),
                ("yarn_beta_slow", C.c_float), ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float),
                ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p), ("type_k", C.c_int), ("type_v", C.c_int),
                ("moe_cache_size", C.c_size_t), ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p),
                ("embeddings", C.c_bool), ("offload_kqv", C.c_bool), ("no_perf", C.c_bool), ("op_offload", C.c_bool),
                ("swa_full", C.c_bool), ("kv_unified", C.c_bool), ("samplers", C.c_void_p),
                ("n_samplers", C.c_size_t), ("ctx_other", C.c_void_p)]


class Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.c_void_p), ("embd", C.c_void_p), ("pos", C.c_void_p),
                ("n_seq_id", C.c_void_p), ("seq_id", C.c_void_p), ("logits", C.c_void_p)]


def bind(lib, name, res, *args):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, list(args)


class Llama:
    """One model and one context of `seqs` sequences of `seq_tokens` tokens. The struct layouts are those of the
    build's include/llama.h."""

    def __init__(self, build, gguf, seqs, seq_tokens, ubatch, threads):
        os.add_dll_directory(build)
        ggml = C.CDLL(os.path.join(build, "ggml.dll"))
        lib = self.lib = C.CDLL(os.path.join(build, "llama.dll"))
        p = C.c_void_p
        bind(ggml, "ggml_backend_load_all_from_path", None, C.c_char_p)
        bind(lib, "llama_backend_init", None)
        bind(lib, "llama_model_default_params", ModelParams)
        bind(lib, "llama_context_default_params", ContextParams)
        bind(lib, "llama_model_load_from_file", p, C.c_char_p, ModelParams)
        bind(lib, "llama_init_from_model", p, p, ContextParams)
        bind(lib, "llama_model_get_vocab", p, p)
        bind(lib, "llama_vocab_n_tokens", C.c_int32, p)
        bind(lib, "llama_tokenize", C.c_int32, p, C.c_char_p, C.c_int32, p, C.c_int32, C.c_bool, C.c_bool)
        bind(lib, "llama_decode", C.c_int32, p, Batch)
        bind(lib, "llama_get_logits_ith", C.POINTER(C.c_float), p, C.c_int32)
        bind(lib, "llama_get_memory", p, p)
        bind(lib, "llama_memory_clear", None, p, C.c_bool)
        # the build loads its backends (CUDA, the CPU variant for this CPU) from its folder, not the process's
        ggml.ggml_backend_load_all_from_path(build.encode())
        lib.llama_backend_init()
        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = 999
        self.model = lib.llama_model_load_from_file(gguf.encode(), mp)
        if not self.model:
            raise SystemExit(f"llama.cpp could not load {gguf}")
        cp = lib.llama_context_default_params()
        cp.n_seq_max, cp.n_ctx, cp.n_batch, cp.n_ubatch = seqs, seqs * seq_tokens, seqs * seq_tokens, ubatch
        cp.n_outputs_max_per_seq = 0   # the default reads only each sequence's last token
        cp.n_threads = cp.n_threads_batch = threads
        self.ctx = lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise SystemExit("llama.cpp could not create the context")
        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)
        self.seqs, self.seq_tokens = seqs, seq_tokens
        n = seqs * seq_tokens
        self.token, self.pos = np.zeros(n, np.int32), np.zeros(n, np.int32)
        self.n_seq_id, self.seq, self.out = np.ones(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int8)
        self.seq_ptr = self.seq.ctypes.data + 4 * np.arange(n, dtype=np.uint64)   # token i's one sequence id
        self.batch = Batch(0, self.token.ctypes.data, None, self.pos.ctypes.data, self.n_seq_id.ctypes.data,
                           self.seq_ptr.ctypes.data, self.out.ctypes.data)

    def tokenize(self, text):
        """llama-server's tokenization of a prompt: special tokens added and parsed."""
        b = text.encode("utf-8")
        buf = np.zeros(len(b) + 8, np.int32)
        n = self.lib.llama_tokenize(self.vocab, b, len(b), buf.ctypes.data, len(buf), True, True)
        return buf[:n].tolist()

    def logits(self, prompts):
        """prompts: up to `seqs` (tokens, positions). Yields per prompt and position its logits, valid until the next
        call."""
        i = 0
        for s, (tokens, positions) in enumerate(prompts):
            if len(tokens) > self.seq_tokens:
                raise SystemExit(f"a prompt of {len(tokens)} tokens does not fit a {self.seq_tokens}-token slot")
            n = len(tokens)
            self.token[i:i + n], self.pos[i:i + n], self.seq[i:i + n] = tokens, np.arange(n), s
            self.out[i:i + n] = 0
            self.out[i + np.array(positions)] = 1
            i += n
        self.batch.n_tokens = i
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        if (err := self.lib.llama_decode(self.ctx, self.batch)) != 0:
            raise SystemExit(f"llama_decode failed with {err}")
        start = 0
        for tokens, positions in prompts:
            for p in positions:
                yield np.ctypeslib.as_array(self.lib.llama_get_logits_ith(self.ctx, start + p), (self.n_vocab,))
            start += len(tokens)


def margin(logits, ban, ok):
    """log P(ban) - log P(ok) as llama-server's top TOP_LOGPROBS logprobs give it (evaluate.classifier_margin)."""
    lb, lo = logits[ban], logits[ok]
    top_b = np.count_nonzero(logits > lb) < TOP_LOGPROBS
    top_o = np.count_nonzero(logits > lo) < TOP_LOGPROBS
    if not (top_b or top_o):
        return None
    floor = 0.0 if top_b and top_o else np.partition(logits, -TOP_LOGPROBS)[-TOP_LOGPROBS]
    return float((lb if top_b else floor) - (lo if top_o else floor))


def main():
    s = json.loads(sys.argv[1])
    llama = Llama(s["build"], s["gguf"], s["seqs"], s["seq_tokens"], s["ubatch"], s["threads"])
    ids = [llama.tokenize(a) for a in (" ban", " ok")]
    if not all(len(i) == 1 for i in ids):
        raise SystemExit(f"the answers are not single tokens in this model: {ids}")
    ban, ok = ids[0][0], ids[1][0]
    windows = []   # (tokens, positions, keys)

    def decode():
        rows = llama.logits([(t, p) for t, p, _ in windows])
        out = [json.dumps([k, margin(next(rows), ban, ok)]) for _, _, keys in windows for k in keys]
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        windows.clear()

    for line in sys.stdin:
        prev = None
        for key, prompt in json.loads(line):
            tokens = llama.tokenize(prompt)
            if not (prev and len(tokens) > len(prev) and tokens[:len(prev)] == prev):
                if len(windows) == llama.seqs:
                    decode()
                windows.append((tokens, [], []))
            windows[-1] = (tokens, windows[-1][1] + [len(tokens) - 1], windows[-1][2] + [key])
            prev = tokens
    if windows:
        decode()


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    main()
