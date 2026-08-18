"""
Generate 20M token corpus (~80MB text) synthetically

We generate diverse domains:
- TinyStories-style children stories (70%)
- Simple QA / factual (10%)
- Math / arithmetic reasoning (10%)
- Python code snippets (5%)
- Dialogues (5%)

Target: 20M tokens => ~80M chars. With vocab 4096, ~4 chars/token avg.
We'll generate ~85M chars to be safe.
"""

import random
import os

random.seed(42)

# word banks
NAMES = ["Lily","Tom","Emma","Sam","Ben","Ava","Mia","Noah","Leo","Lucy","Jack","Sophie","Max","Zoe","Oliver","Ella","Milo","Ruby","Finn","Isla",
         "Charlie","Oscar","Luna","Milo","Harper","Ethan","Aria","Lucas","Grace","Hugo","Ivy","Theo","Nora","Felix","Cora","Arlo","Layla","Jude","Alice"]
PLACES = ["forest","garden","beach","park","school","house","cave","mountain","river","village","castle","meadow","lake","treehouse","library","farm","desert","city","island","playground"]
ADJS = ["happy","little","big","brave","curious","shiny","soft","kind","funny","tiny","magic","quiet","bright","sleepy","silly","clever","gentle","wild","warm","cold","fast","slow","red","blue","golden","silver","ancient","young","old","new","beautiful","little","friendly"]
NOUNS = ["dog","cat","bird","star","tree","flower","book","ball","car","hat","sun","moon","cloud","dragon","robot","fox","bear","bunny","turtle","fish",
         "key","door","box","light","shadow","song","dream","friend","adventure","story","cake","cookie","river","boat","kite","castle","bridge","owl","mouse"]
VERBS = ["found","saw","liked","wanted","went","made","played","ran","jumped","laughed","helped","shared","built","drew","sang","danced","read","wrote","thought","learned","discovered","explored","climbed","opened","closed","carried","gave","took","said"]

STORY_TEMPLATES = [
    "{Name} was a {adj} {noun} who lived near the {place}. One day, {Name} {verb} a {adj2} {noun2}.",
    "Once upon a time, {Name} went to the {place}. There, {Name} met a {adj} {noun} who {verb} very loudly.",
    "{Name} loved to {verb} in the {place}. The {noun} was {adj} and the {noun2} was {adj2}.",
    "It was a {adj} morning. {Name} woke up and {verb} to the {place} with {Name2}.",
    "{Name} and {Name2} were best friends. They {verb} together at the {place} every day. One day they {verb2} a {adj} {noun}.",
    "The {noun} said, \"Hello, {Name}!\" {Name} was surprised. \"How can you talk?\" asked {Name}. The {noun} {verb} and said it was magic.",
    "{Name} had a {noun} that was {adj}. {Name} took it to the {place} to show {Name2}.",
    "In the {place}, there was a {adj} {noun}. {Name} wanted to {verb} it, but it was too {adj2}.",
]

QA_TEMPLATES = [
    "Q: What is the capital of France? A: Paris is the capital of France.\n",
    "Q: How many legs does a dog have? A: A dog has four legs.\n",
    "Q: What color is the sky? A: The sky is blue during the day.\n",
    "Q: What do bees make? A: Bees make honey.\n",
    "Q: Why do plants need sunlight? A: Plants need sunlight to make food through photosynthesis.\n",
]

MATH_TEMPLATES = [
    "Q: What is {a} + {b}? A: {a} + {b} = {c}. Because adding them together gives {c}.\n",
    "Q: If you have {a} apples and get {b} more, how many do you have? A: You have {a}+{b}={c} apples.\n",
    "Problem: {a} * {b} = ? Solution: {a} times {b} equals {c}. You can think of it as adding {a} {b} times.\n",
    "Count: {a}, {b}, {c} ... What comes next? After {c}, if we add {d}, we get {e}.\n",
]

CODE_TEMPLATES = [
    "def add(a, b):\n    return a + b\n\nprint(add({a}, {b})) # prints {c}\n",
    "for i in range({a}):\n    print(f\"Hello {{i}}\")\n",
    "numbers = [{a}, {b}, {c}]\ntotal = sum(numbers)\nprint(f\"Total is {{total}}\")\n",
    "def is_even(n):\n    return n % 2 == 0\n\nprint(is_even({a})) # {even}\n",
    "class {Name}:\n    def __init__(self):\n        self.name = \"{Name}\"\n    def greet(self):\n        return f\"Hi, I am {{self.name}}\"\n",
]

DIALOG_TEMPLATES = [
    "{Name}: Hi {Name2}! How are you?\n{Name2}: I'm {adj}! I just {verb} a {noun} at the {place}.\n{Name}: Wow! That's {adj2}!\n",
    "{Name}: Do you want to {verb} to the {place}?\n{Name2}: Yes! I love the {place}. Let's bring the {adj} {noun}.\n",
]

FACTS = [
    "The sun is a star. It gives light and heat to Earth.",
    "Water can be solid, liquid, or gas. Ice is solid water.",
    "Dogs are friendly animals. They like to play and bark.",
    "The moon goes around the Earth. It looks bright at night.",
    "Trees make oxygen. They have leaves, branches, and roots.",
    "Books have pages with words. Reading helps you learn.",
    "The ocean is very big and has many fish.",
    "Mountains are tall rocks. Snow is often on top.",
    "Stars shine in the night sky. Some are far away suns.",
    "Bicycles have two wheels. You pedal to move.",
]

def random_story():
    t = random.choice(STORY_TEMPLATES)
    return t.format(
        Name=random.choice(NAMES),
        Name2=random.choice(NAMES),
        place=random.choice(PLACES),
        adj=random.choice(ADJS),
        adj2=random.choice(ADJS),
        noun=random.choice(NOUNS),
        noun2=random.choice(NOUNS),
        verb=random.choice(VERBS),
        verb2=random.choice(VERBS),
    ) + " " + random.choice(FACTS) + " " + " ".join([
        f"{random.choice(NAMES)} {random.choice(VERBS)} the {random.choice(ADJS)} {random.choice(NOUNS)}."
        for _ in range(random.randint(1,3))
    ])

def generate_corpus(output_path, target_chars=85_000_000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    story_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        while written < target_chars:
            r = random.random()
            if r < 0.70:
                # story 3-6 sentences
                para = " ".join([random_story() for _ in range(random.randint(2,4))])
                para += "\n\n"
            elif r < 0.80:
                para = random.choice(QA_TEMPLATES*2 + FACTS) + "\n"
            elif r < 0.90:
                a = random.randint(1,100)
                b = random.randint(1,100)
                para = random.choice(MATH_TEMPLATES).format(a=a,b=b,c=a+b,d=random.randint(1,5),e=a+b+random.randint(1,5), even="True" if a%2==0 else "False") + "\n"
            elif r < 0.95:
                a = random.randint(1,20)
                b = random.randint(1,20)
                c = random.randint(1,20)
                para = random.choice(CODE_TEMPLATES).format(a=a,b=b,c=c, even="True" if a%2==0 else "False", Name=random.choice(NAMES)) + "\n"
            else:
                para = random.choice(DIALOG_TEMPLATES).format(
                    Name=random.choice(NAMES), Name2=random.choice(NAMES),
                    adj=random.choice(ADJS), adj2=random.choice(ADJS),
                    noun=random.choice(NOUNS), verb=random.choice(VERBS), place=random.choice(PLACES)
                ) + "\n"

            f.write(para)
            written += len(para)
            story_count += 1
            if story_count % 5000 == 0:
                print(f"Written {written/1e6:.2f}M chars / {target_chars/1e6:.2f}M ({written/target_chars*100:.1f}%) docs={story_count}")

    print(f"DONE: {written} chars written to {output_path}, docs {story_count}")
    size_mb = os.path.getsize(output_path)/1024/1024
    print(f"File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--chars", type=int, default=85_000_000)
    args = ap.parse_args()
    generate_corpus(args.out, args.chars)
