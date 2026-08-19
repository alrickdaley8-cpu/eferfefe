"""
Generate RAG-augmented QA for training model to use retrieved context
Format for training: Context + User + Assistant, where assistant answer is in context
This teaches model to extract answer from context, enabling any question answering via retrieval
"""
import random, json, os
from .retriever import KNOWLEDGE_BASE, retrieve
from .general_qa_data import (
    COUNTRIES, ELEMENTS, PLANETS, SCIENCE_FACTS, HISTORY_FACTS,
    COMMONSENSE, DEFINITIONS, HOWTO, gen_math, gen_geo, gen_element, gen_planet, gen_science, gen_history, gen_commonsense, gen_def, gen_howto, gen_open_ended
)

random.seed(456)

def gen_rag_example():
    # Pick a random QA generator
    generators = [
        (gen_geo, 0.2),
        (gen_element, 0.15),
        (gen_planet, 0.1),
        (gen_science, 0.15),
        (gen_history, 0.1),
        (gen_commonsense, 0.1),
        (gen_def, 0.1),
        (gen_howto, 0.05),
        (gen_open_ended, 0.05),
    ]
    funcs, weights = zip(*generators)
    gen = random.choices(funcs, weights)[0]
    ex = gen()
    # ex has messages: user and assistant
    user_msg = ex["messages"][0]["content"] if ex["messages"][0]["role"]=="user" else ex["messages"][1]["content"]
    assistant_msg = ex["messages"][-1]["content"]

    # Retrieve context for this user query (or use random relevant)
    retrieved = retrieve(user_msg, top_k=2)
    if not retrieved:
        # fallback: pick random knowledge
        retrieved = random.sample(KNOWLEDGE_BASE, 2)

    context = "\n".join(retrieved)

    # Build training example with context included in system or as separate
    # Format: System: You are helpful. Context: ...
    # User: question
    # Assistant: answer

    system_content = f"You are a knowledgeable assistant. Use the following context to answer. Context: {context}"

    # Sometimes include math separately
    if "What is" in user_msg and any(x in user_msg for x in ["+", "-", "*", "%"]):
        # for math, no need RAG, keep simple
        system_content = "You are a helpful math assistant."

    return {
        "messages": [
            {"role":"system","content": system_content},
            {"role":"user","content": user_msg},
            {"role":"assistant","content": assistant_msg}
        ],
        "context": context,
        "query": user_msg
    }

def generate_rag_dataset(num=20000, out_path="data/rag_qa.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for i in range(num):
            ex = gen_rag_example()
            f.write(json.dumps(ex)+"\n")
            if (i+1)%5000==0:
                print(f"Wrote {i+1}/{num}")
    print(f"Done {out_path}")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rag_qa.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    args=ap.parse_args()
    generate_rag_dataset(args.n, args.out)
