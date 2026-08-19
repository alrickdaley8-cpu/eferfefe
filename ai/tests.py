"""Self-contained test suite for the whole tiny-lm stack — no pytest needed.

    python -m ai.tests            # everything
    python -m ai.tests Model      # one group

Tests that need trained artefacts (checkpoints, .bin files) skip cleanly when those are absent,
so this also works on a fresh clone.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, "ai")
DATA = os.path.join(AI, "data")
CKPT = os.path.join(AI, "checkpoints")


def has(path: str) -> bool:
    return os.path.exists(path)


# ======================================================================================
class Model(unittest.TestCase):
    def setUp(self):
        from ai.model import GPT, GPTConfig
        torch.manual_seed(0)
        self.m = GPT(GPTConfig()).eval()

    def test_parameter_count(self):
        self.assertEqual(self.m.num_params(), 5_015_808)
        self.assertEqual(self.m.num_params(non_embedding=True), 2_918_656)

    def test_weight_tying(self):
        self.assertIs(self.m.lm_head.weight, self.m.tok_emb.weight)

    def test_forward_shapes_and_loss(self):
        x = torch.randint(0, 8192, (2, 16))
        logits, loss = self.m(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, 8192))
        self.assertTrue(torch.isfinite(loss))

    def test_kv_cache_matches_full_forward(self):
        x = torch.randint(0, 8192, (1, 48))
        with torch.inference_mode():
            full, _ = self.m(x)
            _, _, cache = self.m(x[:, :-1], use_cache=True)
            step, _, _ = self.m(x[:, -1:], past=cache, use_cache=True)
        self.assertTrue(torch.allclose(full[:, -1], step[:, -1], atol=1e-4))

    def test_last_only_saves_work(self):
        x = torch.randint(0, 8192, (1, 12))
        with torch.inference_mode():
            small, _, _ = self.m(x, use_cache=True, last_only=True)
        self.assertEqual(tuple(small.shape), (1, 1, 8192))

    def test_generation_is_bounded(self):
        out = self.m.generate(torch.randint(0, 8192, (1, 4)), max_new_tokens=6, temperature=0.8)
        self.assertEqual(out.size(1), 10)

    def test_context_limit_is_enforced(self):
        with self.assertRaises(AssertionError):
            self.m(torch.randint(0, 8192, (1, 513)))


# ======================================================================================
class Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(DATA, "tokenizer.json")
        if not has(path):
            raise unittest.SkipTest("tokenizer not built")
        from tokenizers import Tokenizer as T
        cls.tok = T.from_file(path)

    def test_vocab_size(self):
        self.assertEqual(self.tok.get_vocab_size(), 8192)

    def test_eot_token(self):
        self.assertEqual(self.tok.token_to_id("<|endoftext|>"), 0)

    def test_roundtrip(self):
        for text in ("import os\nprint('hello world')", "The quick brown fox — 42%",
                     "pip install black"):
            ids = self.tok.encode(text).ids
            self.assertEqual(self.tok.decode(ids, skip_special_tokens=False), text)

    def test_ids_fit_uint16(self):
        ids = self.tok.encode("a" * 500).ids
        self.assertTrue(max(ids) < 65536)


# ======================================================================================
class Corpus(unittest.TestCase):
    def test_token_budget(self):
        p = os.path.join(DATA, "train.bin")
        if not has(p):
            self.skipTest("train.bin not built")
        self.assertEqual(os.path.getsize(p) // 2, 100_000_000)

    def test_validation_split(self):
        p = os.path.join(DATA, "val.bin")
        if not has(p):
            self.skipTest("val.bin not built")
        self.assertEqual(os.path.getsize(p) // 2, 1_000_000)

    def test_sft_stream_and_mask_align(self):
        t = os.path.join(DATA, "qa", "sft_tokens.bin")
        m = os.path.join(DATA, "qa", "sft_mask.bin")
        if not (has(t) and has(m)):
            self.skipTest("sft bins not built")
        self.assertEqual(os.path.getsize(t) // 2, os.path.getsize(m))
        mask = np.memmap(m, dtype=np.uint8, mode="r")
        frac = float(mask[:2_000_000].mean())
        self.assertTrue(0.05 < frac < 0.6, f"answer-token fraction looks wrong: {frac}")


# ======================================================================================
class Retrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has(os.path.join(DATA, "qa", "knowledge.jsonl")):
            raise unittest.SkipTest("knowledge base not built")
        from ai.retrieve import Retriever
        cls.r = Retriever()

    def test_knowledge_base_loaded(self):
        self.assertGreater(len(self.r.docs), 1000)

    def test_exact_name_wins(self):
        for name in ("numpy", "black", "requests"):
            top = self.r.search(f"tell me about {name}", k=1)[0][1]["name"].lower()
            self.assertEqual(top, name)

    def test_descriptive_query(self):
        names = [d["name"] for _, d in self.r.search("library for progress bars", k=5)]
        self.assertTrue(any("progress" in n or "tqdm" in n for n in names), names)

    def test_context_contains_fields(self):
        ctx = self.r.context_for("what license does black use")
        self.assertIn("License:", ctx)
        self.assertIn("black", ctx.lower())

    def test_low_confidence_returns_nothing(self):
        self.assertIsNone(self.r.context_for("zzzz qqqq wwww vvvv"))


# ======================================================================================
class AnswerLayer(unittest.TestCase):
    def setUp(self):
        from ai import answer
        self.a = answer

    def test_calculator(self):
        cases = [("What is 3471 + 2856?", "6327"), ("What is 47 times 23?", "1081"),
                 ("What is 900 - 250?", "650"), ("What is 25% of 480?", "120"),
                 ("What is the average of 4, 8 and 12?", "8"),
                 ("How many letters are in the word package?", "7"),
                 ("How many times does the letter p appear in pipeline?", "2"),
                 ("How many times does the letter p appear in package?", "1"),
                 ("Reverse the string wheel", "leehw"),
                 ("What comes next: 4, 11, 18, 25?", "32")]
        for q, expect in cases:
            got = self.a.solve(q, [])
            self.assertIsNotNone(got, q)
            self.assertIn(expect, got["answer"], q)

    def test_package_fields(self):
        doc = {"name": "black", "version": "26.5.1", "license": "MIT", "summary": "formatter",
               "description": "", "requires_python": ">=3.10", "author": "Łukasz Langa",
               "deps": ["click", "packaging"], "topics": [], "home": "https://x.dev",
               "keywords": ""}
        checks = [("What license does black use?", "MIT"),
                  ("What is the latest version of black?", "26.5.1"),
                  ("Which Python version does black need?", ">=3.10"),
                  ("Who wrote black?", "Langa"),
                  ("How do I install black?", "pip install black"),
                  ("Does black depend on click?", "Yes"),
                  ("Does black depend on numpy?", "No"),
                  ("How many dependencies does black have?", "2")]
        for q, expect in checks:
            got = self.a.solve(q, [doc])
            self.assertIsNotNone(got, q)
            self.assertIn(expect, got["answer"], q)

    def test_subject_resolution_uses_the_named_package(self):
        a = {"name": "scikit-optimize", "deps": ["numpy", "scipy"], "license": "BSD"}
        b = {"name": "numpy", "deps": [], "license": "BSD"}
        got = self.a.solve("Does scikit-optimize depend on numpy?", [b, a])  # numpy ranked first
        self.assertIn("Yes", got["answer"])

    def test_two_package_comparison(self):
        a = {"name": "alpha", "license": "MIT", "deps": ["x", "y"]}
        b = {"name": "beta", "license": "Apache-2.0", "deps": ["x"]}
        got = self.a.solve("Do alpha and beta use the same license?", [a, b])
        self.assertIn("No", got["answer"])
        got = self.a.solve("Which has more dependencies, alpha or beta?", [a, b])
        self.assertIn("alpha", got["answer"])

    def test_out_of_domain_gets_a_fallback(self):
        out = self.a.verify("Paris is the capital of France.", None, has_context=False)
        self.assertEqual(out["status"], "fallback")

    def test_unsupported_claim_is_rejected(self):
        out = self.a.verify("2098 is maintained by Amelianov.", None, has_context=True,
                            context="black 26.5.1: the uncompromising code formatter. License: MIT.")
        self.assertEqual(out["status"], "fallback")

    def test_grounded_text_always_wins(self):
        truth = {"answer": "requests is released under the Apache-2.0 license.", "steps": [],
                 "kind": "knowledge-base"}
        out = self.a.verify("requests-post is released under the Apache-2.0 license.", truth)
        self.assertEqual(out["final"], truth["answer"])   # hallucinated name never reaches the user

    def test_random_package_is_not_used_for_general_questions(self):
        doc = {"name": "mailman", "summary": "mailing list manager", "license": "GPL",
               "deps": [], "version": "3.0"}
        self.assertIsNone(self.a.solve("What is the capital of France?", [doc]))
        self.assertIsNotNone(self.a.solve("What is mailman?", [doc]))

    def test_conversational_intents(self):
        for q, expect in (("hi", "Hello"), ("who are you", "5,015,808"),
                          ("what can you do", "Python packages"), ("thanks", "welcome"),
                          ("How do I read a text file in Python?", "with open")):
            got = self.a.solve(q, [])
            self.assertIsNotNone(got, q)
            self.assertIn(expect, got["answer"], q)

    def test_degenerate_detection(self):
        for bad in ("- `1.0.0.0.0.0.0.0.0.0", "the the the the the the the", "..;;..;;..;;", ""):
            self.assertTrue(self.a.looks_degenerate(bad), bad)
        for good in ("black is released under the MIT license.",
                     "Yes, fastapi depends on pydantic."):
            self.assertFalse(self.a.looks_degenerate(good), good)

    def test_noise_is_replaced_by_a_fallback(self):
        out = self.a.verify("1.0.0.0.0.0.0.0.0.0", None)
        self.assertEqual(out["status"], "fallback")
        self.assertIn("knowledge base", out["final"])

    def test_verification_statuses(self):
        truth = {"answer": "black is released under the MIT license.", "steps": [],
                 "kind": "knowledge-base"}
        self.assertEqual(self.a.verify("black is released under the MIT license.",
                                       truth)["status"], "ok")
        self.assertEqual(self.a.verify("black uses the Apache license.", truth)["status"],
                         "corrected")
        self.assertEqual(self.a.verify("Beautiful Soup parses broken HTML documents.",
                                       None)["status"], "unchecked")

    def test_numbers_must_match_exactly(self):
        truth = {"answer": "The latest version of x is 0.10.2.", "steps": [], "kind": "kb"}
        self.assertEqual(self.a.verify("The latest version of x is 0.3.2.", truth)["status"],
                         "corrected")

    def test_empty_model_answer_is_replaced(self):
        truth = {"answer": "42", "steps": [], "kind": "calculator"}
        self.assertEqual(self.a.verify("   ", truth)["final"], "42")


# ======================================================================================
class Chat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ai.chat import Assistant, default_ckpt
            default_ckpt()
        except Exception as exc:                      # no checkpoint yet
            raise unittest.SkipTest(f"no servable checkpoint ({exc})")
        cls.a = Assistant(threads=1)

    def test_checkpoint_listing_hides_optimiser_state(self):
        from ai.chat import list_checkpoints
        files = [c["file"] for c in list_checkpoints()]
        self.assertTrue(files)
        self.assertNotIn("ckpt.pt", files)
        self.assertFalse([f for f in files if f.endswith("_ckpt.pt")])

    def test_stream_event_sequence(self):
        kinds = [ev["type"] for ev in self.a.stream("What license does black use?",
                                                    max_tokens=8, temperature=0.2)]
        self.assertEqual(kinds[0], "thought")
        self.assertEqual(kinds[-1], "done")

    def test_done_payload(self):
        done = [ev for ev in self.a.stream("What license does black use?", max_tokens=8,
                                           temperature=0.2) if ev["type"] == "done"][0]
        for key in ("answer", "reasoning", "verification", "stats", "checkpoint"):
            self.assertIn(key, done)
        self.assertIn(done["verification"], ("ok", "corrected", "unchecked"))

    def test_think_tags_never_leak(self):
        for ev in self.a.stream("What license does black use?", max_tokens=24, temperature=0.2):
            if ev["type"] == "token":
                self.assertNotIn("<think", ev["text"])
                self.assertNotIn("</think", ev["text"])
                self.assertIn(ev["channel"], ("think", "answer"))

    def test_arithmetic_skips_retrieval(self):
        evs = list(self.a.stream("What is 12 + 30?", max_tokens=8, temperature=0.2))
        first = evs[0]["text"].lower()
        self.assertIn("without retrieval", first)

    def test_grounded_answer_is_correct(self):
        done = [ev for ev in self.a.stream("What is 3471 + 2856?", max_tokens=40,
                                           temperature=0.2) if ev["type"] == "done"][0]
        self.assertIn("6327", done["answer"])


# ======================================================================================
class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from ai.chat import default_ckpt
            default_ckpt()
        except Exception as exc:
            raise unittest.SkipTest(f"no servable checkpoint ({exc})")
        from ai import serve
        serve.get_assistant()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return r.status, r.read().decode()

    def post(self, path: str, payload: dict):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()

    def test_landing_page(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("tiny-lm", body)
        self.assertIn("chatapp", body)          # the embedded chat widget

    def test_app_view(self):
        status, body = self.get("/app")
        self.assertEqual(status, 200)
        self.assertIn("composer", body)

    def test_models_endpoint(self):
        status, body = self.get("/models")
        d = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(d["params"], 5_015_808)
        self.assertTrue(any(m["loaded"] for m in d["models"]))

    def test_status_endpoint(self):
        status, body = self.get("/status")
        self.assertEqual(status, 200)
        self.assertIn("state", json.loads(body))

    def test_chat_endpoint(self):
        status, body = self.post("/chat", {"message": "What is 12 + 30?", "tokens": 24,
                                           "temperature": 0.2})
        d = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("42", d["answer"])

    def test_stream_endpoint(self):
        status, body = self.post("/chat/stream", {"message": "What license does black use?",
                                                  "tokens": 24, "temperature": 0.2})
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(any(e["type"] == "token" for e in events))

    def test_unknown_route(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/nope")


# ======================================================================================
class Pipeline(unittest.TestCase):
    def test_daemon_script_syntax(self):
        rc = subprocess.run(["bash", "-n", os.path.join(AI, "daemon.sh")]).returncode
        self.assertEqual(rc, 0)

    def test_status_file_is_valid(self):
        p = os.path.join(CKPT, "status.json")
        if not has(p):
            self.skipTest("no training status yet")
        s = json.load(open(p))
        self.assertIn(s["state"], ("running", "paused", "done"))
        self.assertLessEqual(s.get("tokens", 0), s.get("total_tokens", 1) + 1)

    def test_page_renders(self):
        from ai.make_page import render
        html = render({"pre": [{"tok": 1e6, "ema": 4.0}, {"tok": 2e6, "ema": 3.5}],
                       "sft": [], "status": {"tokens": 2e6, "total_tokens": 1e8},
                       "kb": 100, "ckpts": []}, [])
        self.assertIn("<svg", html)
        self.assertIn("Talk to it", html)
        self.assertIn("5,015,808", html)

    def test_eval_scoring(self):
        from ai.eval import scored
        self.assertTrue(scored("black is released under the MIT license.", "MIT"))
        self.assertTrue(scored("Yes, x depends on y.", "yes"))
        self.assertFalse(scored("No, x does not depend on y.", "yes"))
        self.assertFalse(scored("version 0.3.2", "0.10.2"))

    def test_chat_format_roundtrip(self):
        from ai.finetune import format_prompt
        p = format_prompt("hello?")
        self.assertTrue(p.startswith("### Question:\n"))
        self.assertTrue(p.endswith("### Answer:\n"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2, argv=[sys.argv[0]] + sys.argv[1:])
