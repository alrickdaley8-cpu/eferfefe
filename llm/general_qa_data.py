"""
Generate broad general knowledge QA for 1M model to answer any question
Covers: geography, science, history, math, commonsense, definitions, how-to, etc.
Target: 50k examples, ~2-3M tokens, to fine-tune base model
"""
import random, json, os
random.seed(123)

# Knowledge bases
COUNTRIES = [
    ("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("USA","Washington D.C."),("UK","London"),
    ("Canada","Ottawa"),("Australia","Canberra"),("Brazil","Brasilia"),("India","New Delhi"),("China","Beijing"),
    ("Russia","Moscow"),("Italy","Rome"),("Spain","Madrid"),("Mexico","Mexico City"),("Egypt","Cairo"),
    ("South Korea","Seoul"),("North Korea","Pyongyang"),("South Africa","Pretoria"),("Nigeria","Abuja"),
    ("Argentina","Buenos Aires"),("Chile","Santiago"),("Peru","Lima"),("Colombia","Bogota"),("Sweden","Stockholm"),
    ("Norway","Oslo"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Poland","Warsaw"),("Turkey","Ankara"),
    ("Greece","Athens"),("Portugal","Lisbon"),("Netherlands","Amsterdam"),("Belgium","Brussels"),
    ("Switzerland","Bern"),("Austria","Vienna"),("Ireland","Dublin"),("New Zealand","Wellington"),
    ("Thailand","Bangkok"),("Vietnam","Hanoi"),("Indonesia","Jakarta"),("Pakistan","Islamabad"),
    ("Bangladesh","Dhaka"),("Saudi Arabia","Riyadh"),("Iran","Tehran"),("Iraq","Baghdad"),
]

ELEMENTS = [
    ("Hydrogen","H",1),("Helium","He",2),("Lithium","Li",3),("Carbon","C",6),("Nitrogen","N",7),
    ("Oxygen","O",8),("Sodium","Na",11),("Magnesium","Mg",12),("Aluminum","Al",13),("Silicon","Si",14),
    ("Iron","Fe",26),("Copper","Cu",29),("Silver","Ag",47),("Gold","Au",79),("Mercury","Hg",80),
    ("Lead","Pb",82),("Uranium","U",92)
]

PLANETS = [
    ("Mercury","closest to the sun, smallest planet"),
    ("Venus","hottest planet, thick atmosphere"),
    ("Earth","only planet known to support life, 71% water"),
    ("Mars","red planet, has Olympus Mons largest volcano"),
    ("Jupiter","largest planet, gas giant, Great Red Spot"),
    ("Saturn","famous for its rings, gas giant"),
    ("Uranus","ice giant, tilted on its side"),
    ("Neptune","ice giant, farthest from sun, strong winds"),
]

SCIENCE_FACTS = [
    ("What is the largest organ in the human body?", "The skin is the largest organ in the human body."),
    ("How many bones are in the human body?", "An adult human body has 206 bones."),
    ("What is the speed of light?", "The speed of light in vacuum is about 299,792 kilometers per second."),
    ("What is photosynthesis?", "Photosynthesis is how plants make food using sunlight, carbon dioxide, and water, producing oxygen."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("What gas do humans breathe in?", "Humans breathe in oxygen (O2)."),
    ("What gas do plants breathe in?", "Plants take in carbon dioxide (CO2) for photosynthesis."),
]

HISTORY_FACTS = [
    ("When did World War 2 end?", "World War 2 ended in 1945."),
    ("When did humans first land on the moon?", "Humans first landed on the moon in 1969 with Apollo 11."),
    ("Who invented the light bulb?", "Thomas Edison is credited with inventing the practical light bulb in 1879."),
    ("When did the Berlin Wall fall?", "The Berlin Wall fell in 1989."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
]

MATH_TEMPLATES = [
    ("What is {a} + {b}?", "{a} + {b} = {c}. Adding them gives {c}."),
    ("What is {a} * {b}?", "{a} * {b} = {c}. Multiplying gives {c}."),
    ("What is {a} - {b}?", "{a} - {b} = {c}. Subtracting gives {c}."),
    ("What is {a}% of {b}?", "{a}% of {b} is {c}. You calculate {a}/100 * {b} = {c}."),
    ("If you have {a} apples and eat {b}, how many left?", "You have {a} - {b} = {c} apples left."),
]

COMMONSENSE = [
    ("Why do we wear jackets in winter?", "We wear jackets in winter to stay warm because it's cold outside."),
    ("Why do birds fly south in winter?", "Birds fly south in winter to find warmer weather and more food."),
    ("What do you use to write on paper?", "You use a pen or pencil to write on paper."),
    ("Why do we need to drink water?", "We need to drink water to stay hydrated and keep our body working properly."),
    ("What makes day and night?", "Day and night happen because Earth rotates on its axis. When your side faces the sun it's day, when away it's night."),
]

DEFINITIONS = [
    ("What is a computer?", "A computer is an electronic device that processes data and performs tasks following instructions."),
    ("What is gravity?", "Gravity is a force that pulls objects toward each other. Earth's gravity keeps us on the ground."),
    ("What is an algorithm?", "An algorithm is a step-by-step set of instructions to solve a problem, like a recipe."),
    ("What is AI?", "AI stands for Artificial Intelligence. It's when computers are made to think or learn like humans."),
    ("What is the internet?", "The internet is a global network connecting computers worldwide to share information."),
]

HOWTO = [
    ("How do you make a paper airplane?", "To make a paper airplane: 1. Take a paper, fold in half lengthwise. 2. Unfold and fold top corners to center. 3. Fold again to center. 4. Fold in half and make wings."),
    ("How do you boil an egg?", "To boil an egg: 1. Put eggs in pot with water. 2. Boil 10 minutes. 3. Cool in cold water. 4. Peel and eat."),
    ("How to save money?", "To save money: 1. Track what you spend. 2. Make a budget. 3. Avoid unnecessary buys. 4. Put some money aside each month."),
]

def gen_geo():
    country, capital = random.choice(COUNTRIES)
    templates = [
        (f"What is the capital of {country}?", f"The capital of {country} is {capital}."),
        (f"Which country has {capital} as its capital?", f"{country} has {capital} as its capital."),
        (f"Tell me about {country}.", f"{country} is a country whose capital is {capital}."),
    ]
    q,a = random.choice(templates)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_element():
    name, sym, num = random.choice(ELEMENTS)
    templates = [
        (f"What is the chemical symbol for {name}?", f"The chemical symbol for {name} is {sym}."),
        (f"What element has symbol {sym}?", f"The element with symbol {sym} is {name}."),
        (f"What is the atomic number of {name}?", f"The atomic number of {name} is {num}."),
    ]
    q,a = random.choice(templates)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_planet():
    name, desc = random.choice(PLANETS)
    templates = [
        (f"Tell me about {name}.", f"{name} is a planet in our solar system. It is {desc}."),
        (f"What is special about {name}?", f"{name} is {desc}."),
        (f"How many planets are in the solar system?", "There are 8 planets in the solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune."),
    ]
    q,a = random.choice(templates)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_science():
    q,a = random.choice(SCIENCE_FACTS)
    # paraphrase
    variants = [q, f"Explain: {q}", f"Can you tell me {q.lower()}"]
    return {"messages":[{"role":"user","content":random.choice(variants)},{"role":"assistant","content":a}]}

def gen_history():
    q,a = random.choice(HISTORY_FACTS)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_math():
    a = random.randint(1,100)
    b = random.randint(1,100)
    if a<b and random.random()<0.5:
        a,b = b,a
    tmpl_q, tmpl_a = random.choice(MATH_TEMPLATES)
    # handle special cases
    if "{c}" in tmpl_q or "{c}" in tmpl_a:
        if "+" in tmpl_q:
            c=a+b
        elif "*" in tmpl_q:
            # keep small
            a=random.randint(1,12); b=random.randint(1,12); c=a*b
        elif "-" in tmpl_q:
            c=a-b
        elif "%" in tmpl_q:
            a=random.choice([10,15,20,25,50]); b=random.randint(1,200); c = int(a/100*b)
        else:
            c=a+b  # fallback
        q = tmpl_q.format(a=a,b=b,c=c)
        ans = tmpl_a.format(a=a,b=b,c=c)
    else:
        q = tmpl_q.format(a=a,b=b)
        ans = tmpl_a.format(a=a,b=b,c=a+b)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":ans}]}

def gen_commonsense():
    q,a = random.choice(COMMONSENSE)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_def():
    q,a = random.choice(DEFINITIONS)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_howto():
    q,a = random.choice(HOWTO)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

def gen_open_ended():
    prompts = [
        ("What is the meaning of life?", "Many people think the meaning of life is to be happy, help others, learn, and make the world better. Different cultures have different answers."),
        ("How can I be happy?", "To be happy: spend time with loved ones, do things you enjoy, help others, stay healthy with exercise and sleep, and be grateful."),
        ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
        ("What is love?", "Love is a deep feeling of care and connection for someone or something. It makes you want to help and be close to them."),
        ("How to learn programming?", "To learn programming: 1. Pick a language like Python. 2. Do small exercises daily. 3. Build projects you like. 4. Read others' code. 5. Practice regularly."),
        ("What is the best way to study?", "Best way to study: 1. Make a plan. 2. Study in short focused sessions. 3. Take notes. 4. Test yourself. 5. Rest well."),
    ]
    q,a = random.choice(prompts)
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

GENERATORS = [
    (gen_geo, 0.15),
    (gen_element, 0.1),
    (gen_planet, 0.05),
    (gen_science, 0.15),
    (gen_history, 0.08),
    (gen_math, 0.15),
    (gen_commonsense, 0.08),
    (gen_def, 0.08),
    (gen_howto, 0.05),
    (gen_open_ended, 0.11),
]

def generate_general_qa(num=50000, out_path="data/general_qa.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    funcs, weights = zip(*GENERATORS)
    with open(out_path, 'w') as f:
        for i in range(num):
            gen = random.choices(funcs, weights)[0]
            ex = gen()
            # sometimes add system prompt
            if random.random()<0.3:
                ex["messages"] = [{"role":"system","content":"You are TinyLLM, a knowledgeable assistant that can answer any question concisely and accurately."}] + ex["messages"]
            f.write(json.dumps(ex)+"\n")
            if (i+1)%5000==0:
                print(f"Wrote {i+1}/{num}")
    print(f"Done {out_path}")

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="data/general_qa.jsonl")
    ap.add_argument("--n", type=int, default=50000)
    args=ap.parse_args()
    generate_general_qa(args.n, args.out)
