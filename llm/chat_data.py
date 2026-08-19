"""
Generate chat instruction fine-tuning dataset for TinyLLM
Makes model more ChatGPT-like after pretraining
"""
import random, json, os
random.seed(42)

# Same banks as before but for instruction style
NAMES = ["Lily","Tom","Emma","Sam","Ben","Ava","Mia","Noah","Leo","Lucy","Jack","Sophie","Max","Zoe","Oliver","Ella","Milo","Ruby","Finn","Isla"]
PLACES = ["forest","garden","beach","park","school","house","cave","mountain","river","village","castle","meadow","lake","treehouse","library","farm"]
ADJS = ["happy","brave","curious","shiny","kind","funny","magic","bright","clever","gentle","beautiful","little"]
NOUNS = ["dog","cat","bird","star","tree","flower","dragon","robot","fox","bear","bunny","key","door","book","kite"]

def gen_story_instruction():
    name = random.choice(NAMES)
    place = random.choice(PLACES)
    noun = random.choice(NOUNS)
    adj = random.choice(ADJS)
    prompts = [
        f"Tell me a story about {name}.",
        f"Write a short story about a {adj} {noun}.",
        f"Can you tell me a story set in the {place}?",
        f"Once upon a time, {name} went to the {place}. What happened next?",
    ]
    user = random.choice(prompts)
    # generate a short story (2-3 sentences) using template
    story = f"Once upon a time, {name} went to the {place}. There, {name} found a {adj} {noun}. The {noun} was {adj} and it helped {name} have a wonderful adventure. The end."
    return {"messages": [{"role":"user","content":user},{"role":"assistant","content":story}]}

def gen_qa_instruction():
    qas = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How many legs does a dog have?", "A dog has four legs."),
        ("What do bees make?", "Bees make honey."),
        ("Why is the sky blue?", "The sky looks blue because sunlight scatters in the atmosphere. Blue light scatters more than other colors."),
        ("What is photosynthesis?", "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide."),
        ("Tell me a fun fact about dogs.", "Dogs have a sense of smell that is 40 times better than humans! They also wag their tails when happy."),
    ]
    q,a = random.choice(qas)
    # maybe paraphrase user
    user_variants = [q, f"Can you tell me {q.lower()}", f"{q} Explain simply."]
    return {"messages": [{"role":"user","content":random.choice(user_variants)},{"role":"assistant","content":a}]}

def gen_math_instruction():
    a = random.randint(1,50)
    b = random.randint(1,50)
    op = random.choice(['+','-','*'])
    if op=='+':
        c=a+b
        user = random.choice([f"What is {a} + {b}?", f"Calculate {a} + {b}", f"Q: {a} + {b} = ?"])
        assistant = f"{a} + {b} = {c}. I added {a} and {b} together to get {c}."
    elif op=='-':
        if a<b: a,b=b,a
        c=a-b
        user = f"What is {a} - {b}?"
        assistant = f"{a} - {b} = {c}. If you have {a} and take away {b}, you have {c} left."
    else:
        a=random.randint(1,12)
        b=random.randint(1,12)
        c=a*b
        user = f"What is {a} * {b}?"
        assistant = f"{a} * {b} = {c}. That's {a} times {b} equals {c}."
    return {"messages": [{"role":"user","content":user},{"role":"assistant","content":assistant}]}

def gen_code_instruction():
    templates = [
        ("Write a Python function to add two numbers.", "def add(a, b):\n    return a + b\n\n# Example\nprint(add(2, 3))  # 5"),
        ("How do I print hello world in Python?", "In Python, you can print with:\n\n```python\nprint(\"Hello, world!\")\n```"),
        ("Write a loop that prints numbers 1 to 5.", "```python\nfor i in range(1, 6):\n    print(i)\n```\nThis will print 1, 2, 3, 4, 5."),
        ("What is a variable in programming?", "A variable is like a box that stores information. For example: `x = 5` stores the number 5 in a box named x."),
    ]
    u,a = random.choice(templates)
    return {"messages": [{"role":"user","content":u},{"role":"assistant","content":a}]}

def gen_chat_instruction():
    dialogues = [
        ("Hello!", "Hello! How can I help you today?"),
        ("How are you?", "I'm just a tiny AI, but I'm doing great! I'm a 1M parameter model trained on 20M tokens. How can I help you?"),
        ("Who are you?", "I am TinyLLM, a small language model with 1 million parameters trained on 20 million tokens. I was built from scratch to be helpful and friendly!"),
        ("What can you do?", "I can tell stories, answer simple questions, do basic math, and write short Python code. I'm tiny (4MB) so I run on CPU!"),
    ]
    u,a = random.choice(dialogues)
    return {"messages": [{"role":"user","content":u},{"role":"assistant","content":a}]}

def generate_chat_dataset(num_examples=10000, out_path="data/chat_finetune.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generators = [gen_story_instruction, gen_qa_instruction, gen_math_instruction, gen_code_instruction, gen_chat_instruction]
    weights = [0.3, 0.2, 0.2, 0.15, 0.15]
    with open(out_path, 'w') as f:
        for i in range(num_examples):
            gen = random.choices(generators, weights)[0]
            ex = gen()
            # add system prompt sometimes
            if random.random()<0.3:
                ex["messages"] = [{"role":"system","content":"You are TinyLLM, a helpful, friendly, concise AI."}] + ex["messages"]
            f.write(json.dumps(ex)+"\n")
            if (i+1)%1000==0:
                print(f"Wrote {i+1}/{num_examples}")
    print(f"Done: {out_path} {num_examples} examples")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chat_finetune.jsonl")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    generate_chat_dataset(args.n, args.out)
