"""
Generate in-depth QA dataset from deep_knowledge for making model explain things in depth
"""
import random, json, os
from .deep_knowledge import DEEP_KNOWLEDGE
from .retriever_v2 import FULL_KB
import re

random.seed(789)

def generate_deep_qa(num=10000, out_path="data/deep_qa.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Create variations of questions for each deep knowledge entry
    question_templates = [
        "What is {topic}? Explain in depth.",
        "Explain {topic} in detail, step by step.",
        "Tell me everything about {topic} in depth.",
        "Can you explain {topic} thoroughly with examples?",
        "What is {topic}? Give an in-depth explanation.",
        "Explain {topic} like I'm curious and want to understand deeply.",
        "Why is {topic} important? Explain in depth.",
        "How does {topic} work? Explain in depth with steps.",
    ]

    # Also generate specific questions from keywords
    specific_q_templates = [
        "What is {kw}?",
        "Explain {kw} in depth.",
        "Tell me about {kw}.",
        "How does {kw} work? Explain in detail.",
        "Why is {kw} important?",
    ]

    with open(out_path, 'w') as f:
        for i in range(num):
            entry = random.choice(DEEP_KNOWLEDGE)
            topic = entry["topic"]
            # Pick a keyword from topic
            kw = random.choice(entry["keywords"]) if entry["keywords"] else topic.split()[0]

            # Choose template
            if random.random() < 0.5:
                tmpl = random.choice(question_templates)
                q = tmpl.format(topic=topic)
            else:
                tmpl = random.choice(specific_q_templates)
                q = tmpl.format(kw=kw, topic=topic)

            # Answer is deep explanation, but sometimes truncated to fit 256 context
            deep_answer = entry["deep"]
            # Keep answer under 400 tokens (~1600 chars) to fit context
            if len(deep_answer) > 1200:
                deep_answer = deep_answer[:1200] + "..."

            # Add optional system prompt with context for RAG training
            if random.random() < 0.4:
                # Include context from same entry's short + part of deep
                context = entry["short"] + " " + " ".join(FULL_KB[:2])[:200]
                system = f"You are a knowledgeable assistant that explains things in depth, step-by-step with examples. Use context: {context}"
            else:
                system = "You are a knowledgeable assistant that explains things in depth, step by step, with examples, why it matters, and fun facts."

            ex = {
                "messages": [
                    {"role":"system","content": system},
                    {"role":"user","content": q},
                    {"role":"assistant","content": deep_answer}
                ],
                "topic": topic
            }
            f.write(json.dumps(ex)+"\n")
            if (i+1)%2000==0:
                print(f"Wrote {i+1}/{num}")

    print(f"Done {out_path} {num} deep QA examples")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="data/deep_qa.jsonl")
    ap.add_argument("--n", type=int, default=10000)
    args=ap.parse_args()
    generate_deep_qa(args.n, args.out)
