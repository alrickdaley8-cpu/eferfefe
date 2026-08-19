"""
Deep knowledge base for in-depth explanations
Each entry has short fact + detailed in-depth explanation
Used to make 1M model explain things more in depth
"""

DEEP_KNOWLEDGE = [
    {
        "topic": "Paris France capital",
        "keywords": ["france", "paris", "capital"],
        "short": "France's capital is Paris. France is in Europe.",
        "deep": """Paris is the capital and largest city of France, located in the north-central part of the country along the Seine River.

In-depth explanation:
- Geography: Paris sits in the Île-de-France region, built on both banks of the Seine. The city proper has about 2.1 million people, but the metro area has 12 million, making it one of Europe's largest.
- History: Founded over 2,000 years ago as Lutetia by the Parisii tribe, it became capital in 508 AD under Clovis I. It has been center of French politics, art, and revolution.
- Culture & Landmarks: Eiffel Tower (built 1889 for World's Fair, 330m tall), Louvre Museum (world's largest art museum, Mona Lisa), Notre-Dame Cathedral (Gothic, 1163-1344), Champs-Élysées avenue, Montmartre.
- Why it matters: Paris is global center for fashion, gastronomy, art, and diplomacy. 30+ million tourists yearly.
- Fun fact: Called City of Light because early street lighting and Enlightenment center."""
    },
    {
        "topic": "Photosynthesis",
        "keywords": ["photosynthesis", "plant", "sunlight"],
        "short": "Photosynthesis is how plants make food using sunlight, CO2, and water.",
        "deep": """Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy.

In-depth step-by-step:
1. Light Absorption: Chlorophyll in chloroplasts (mostly in leaves) absorbs sunlight, mainly blue and red wavelengths.
2. Water Splitting: Light energy splits water molecules (H2O) into hydrogen, oxygen, and electrons. Oxygen is released as byproduct we breathe.
3. Carbon Fixation (Calvin Cycle): CO2 from air enters through stomata, combines with hydrogen using ATP energy from light, forming glucose (C6H12O6).
4. Equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

Why it matters: 
- Basis of most food chains - all animals depend on plants
- Produces oxygen we breathe (about 50% from ocean phytoplankton)
- Removes CO2, helps climate
- Stores solar energy as chemical energy

Examples: A tree's leaves are solar panels. In 1 hour, a large tree can produce enough oxygen for 2 people for a day."""
    },
    {
        "topic": "Gold chemical symbol",
        "keywords": ["gold", "au", "chemical", "symbol", "element"],
        "short": "Gold's chemical symbol is Au, atomic number 79.",
        "deep": """Gold (Au) is a chemical element and precious metal.

In-depth details:
- Symbol: Au comes from Latin 'Aurum' meaning shining dawn.
- Atomic number: 79, meaning 79 protons. Atomic weight ~197.
- Properties: Highly malleable (1 gram can be beaten into 1 sq meter sheet), ductile, doesn't tarnish or corrode, excellent conductor, dense (19.3x water), yellow color, melts at 1064°C.
- History: Used by humans for 6,000+ years. First coins 700 BC Lydia. Basis of monetary systems until 1971.
- Where found: Formed in supernovae and neutron star collisions. Mined in South Africa, China, Australia, Russia. All gold ever mined would fit in 22m cube.
- Uses: 50% jewelry, 40% investment, 10% electronics (phones have ~0.03g gold for reliable connections), dentistry, aerospace.
- Why valuable: Rare (0.003 ppm in crust), beautiful, doesn't decay, divisible, universally desired.

Fun fact: Olympic gold medals are only 1% gold, mostly silver."""
    },
    {
        "topic": "Solar system planets",
        "keywords": ["planets", "solar system", "how many"],
        "short": "There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        "deep": """Our solar system has 8 planets orbiting the Sun, formed 4.6 billion years ago.

In-depth tour:
1. Mercury: Smallest (4,879 km diameter), closest to Sun (58M km), no atmosphere, extreme temps -173 to 427°C, 88-day year.
2. Venus: Hottest 462°C due to thick CO2 atmosphere, pressure 92x Earth, rotates backwards, day longer than year.
3. Earth: Only life known, 71% water, atmosphere 78% nitrogen 21% oxygen, 1 moon, diameter 12,742 km, 365-day orbit.
4. Mars: Red from iron oxide, half Earth size, thin CO2 atmosphere, -65°C avg, has Olympus Mons largest volcano (22km high) and Valles Marineris canyon, 2 moons.
5. Jupiter: Largest (139k km diameter), gas giant 318x Earth mass, Great Red Spot storm 400 years old bigger than Earth, 95 moons including Ganymede largest moon in system.
6. Saturn: Famous rings made of ice and rock, density less than water (would float!), 146 moons, Titan has thick atmosphere and lakes.
7. Uranus: Ice giant, pale blue from methane, tilted 98° so orbits on side, -195°C, 28 moons.
8. Neptune: Farthest, deep blue, strongest winds 2,000 km/h, -201°C, 16 moons, Triton orbits backward.

Plus dwarf planets like Pluto (demoted 2006), asteroid belt between Mars-Jupiter, Kuiper belt beyond Neptune.

Why 8 not 9? Pluto reclassified 2006 because hasn't cleared orbit."""
    },
    {
        "topic": "World War 2 end",
        "keywords": ["world war 2", "ww2", "ended", "when"],
        "short": "World War 2 ended in 1945.",
        "deep": """World War II (1939-1945) was deadliest conflict in history, 70-85 million deaths.

In-depth timeline:
- Start: Sept 1 1939 Germany invades Poland, UK/France declare war.
- Axis: Germany, Italy, Japan. Allies: USA, UK, USSR, France, China etc.
- Major events: Blitzkrieg, Battle of Britain 1940, Pearl Harbor Dec 7 1941 brings USA in, D-Day June 6 1944 Allied invasion of France, atomic bombs Aug 6 & 9 1945 on Hiroshima Nagasaki.
- End in Europe: May 7-8 1945 Germany surrenders (VE Day).
- End in Pacific: Aug 15 1945 Japan surrenders after atomic bombs, formal Sept 2 1945 (VJ Day).
- Aftermath: UN founded 1945 to prevent future wars, Cold War begins USA vs USSR, decolonization, Nuremberg trials.

Why it matters: Reshaped world order, led to human rights, EU, and awareness of genocide (Holocaust 6M Jews killed).

Fun fact: Code-breaking at Bletchley Park (Alan Turing) shortened war by 2 years."""
    },
    {
        "topic": "Largest organ human body",
        "keywords": ["largest organ", "human body", "skin"],
        "short": "Skin is largest organ.",
        "deep": """The skin is the largest organ in human body by both weight and surface area.

In-depth details:
- Size: About 2 square meters (22 sq ft), weighs ~3.6kg (8 lbs) for adult, 16% of body weight.
- Layers: Epidermis (outer, waterproof, makes new cells), Dermis (middle, contains sweat glands, hair follicles, blood vessels, nerve endings), Hypodermis (fat layer, insulation).
- Functions:
  1. Protection: Barrier against bacteria, UV, chemicals, water loss.
  2. Sensation: Millions of nerve endings detect touch, pressure, pain, temperature.
  3. Temperature regulation: Sweat cools, blood vessels dilate/constrict.
  4. Vitamin D production: Sunlight converts cholesterol to vitamin D.
  5. Immune: Contains immune cells.
- Renewal: Completely renews every 27 days, you shed 500M skin cells daily!
- Comparison: Second largest is liver (1.5kg), then brain (1.4kg).

Why it matters: Without skin, we'd dehydrate and get infected instantly.

Care tips: Sunscreen prevents UV damage, moisturize, stay hydrated."""
    },
    {
        "topic": "What is AI",
        "keywords": ["ai", "artificial intelligence", "what is"],
        "short": "AI is Artificial Intelligence, computers thinking like humans.",
        "deep": """Artificial Intelligence (AI) is when computers are made to perform tasks that normally require human intelligence.

In-depth breakdown:
- Definition: AI is broad field creating systems that can learn, reason, perceive, understand language, and make decisions.
- Types:
  1. Narrow AI (today): Good at one task, e.g., ChatGPT for text, Self-driving for driving, AlphaGo for Go. This is all current AI.
  2. General AI (AGI, future): Human-level across all tasks, can reason generally. Not yet achieved.
  3. Super AI: Smarter than humans in all ways. Sci-fi currently.
- How it works: Machine Learning is subset of AI where computers learn from data without explicit programming. Deep Learning is subset of ML using neural networks with many layers (like our TinyLLM with 6 layers).
  - Training: Show model millions examples, it adjusts internal weights (1M in our case, 175B in GPT-3) to minimize error.
  - Inference: Uses learned weights to predict next word, classify image, etc.
- Examples you use: Siri, Alexa, Google Translate, Netflix recommendations, spam filter, face unlock.
- TinyLLM: Our model is 1M params (vs GPT-4 1.7T). Trained on 20M tokens (vs trillions). Runs on CPU 4MB. Shows core principles.

Why it matters: Transforming medicine (drug discovery), science, education, but raises concerns about jobs, bias, privacy.

Future: Will augment humans, not replace. Need responsible development."""
    },
    {
        "topic": "How many bones human",
        "keywords": ["bones", "human body", "how many"],
        "short": "Adult human has 206 bones.",
        "deep": """Adult human skeleton has 206 bones, but babies have 270.

In-depth explanation:
- Why fewer as adult? Some bones fuse together as we grow. Example: skull starts as several plates that fuse, sacrum fuses from 5 vertebrae.
- Categories:
  - Axial: 80 bones (skull 22, spine 33 vertebrae, ribs 12 pairs, sternum)
  - Appendicular: 126 bones (arms 64, legs 62 including pelvis)
- Smallest: Stapes in ear, 2.8mm.
- Largest: Femur thigh bone, 25% of height.
- Composition: 30% organic (collagen makes flexible), 70% inorganic (calcium phosphate makes hard). Living tissue with blood supply.
- Functions: Support, protection (skull protects brain, ribs heart/lungs), movement (muscles pull bones), blood cell production in marrow, mineral storage.
- Fun facts: Half of bones are in hands and feet (106). Bones are 5x stronger than steel same weight. Heaviest bone disease can make bones 8x normal weight."""
    },
    {
        "topic": "Water formula",
        "keywords": ["water", "formula", "h2o"],
        "short": "Water formula is H2O.",
        "deep": """Water's chemical formula is H2O: 2 hydrogen atoms bonded to 1 oxygen atom.

In-depth science:
- Structure: Bent shape 104.5° angle, polar molecule (oxygen slightly negative, hydrogen positive). This polarity makes water unique.
- Why H2O not HO? Oxygen needs 2 electrons to fill outer shell, hydrogen provides 1 each, so 2 H per O.
- Properties from structure:
  - High surface tension: Water striders walk on water
  - High boiling point: 100°C vs similar molecules like H2S -60°C
  - Expands when freezes: Ice less dense than water, floats, insulates lakes, life survives winter
  - Excellent solvent: Dissolves many substances, transports nutrients in blood
- States: Solid ice (0°C), liquid water, gas steam/water vapor (100°C). Only substance naturally all three on Earth.
- Importance: Covers 71% Earth, 60% human body, 90% blood. No known life without water. Used in photosynthesis, temperature regulation (high specific heat).

Cycle: Evaporation → clouds → rain → rivers → ocean, powered by sun.

Fun fact: 1 liter water has 33,000,000,000,000,000,000,000 molecules."""
    },
]

def get_deep_knowledge(query, top_k=1):
    """Retrieve deep knowledge for any question"""
    import re
    q_lower = query.lower()
    keywords = re.findall(r'\b\w{3,}\b', q_lower)
    scored = []
    for entry in DEEP_KNOWLEDGE:
        score = 0
        for kw in entry["keywords"]:
            if kw in q_lower:
                score += 2
        for kw in keywords:
            if any(kw in k for k in entry["keywords"]):
                score += 1
            if kw in entry["short"].lower() or kw in entry["topic"].lower():
                score += 0.5
        if score>0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for s,e in scored[:top_k]]

if __name__ == "__main__":
    tests = ["capital of France", "photosynthesis", "gold symbol", "how many planets", "largest organ", "what is AI"]
    for q in tests:
        res = get_deep_knowledge(q, top_k=1)
        print(f"Q: {q}\nDeep: {res[0]['deep'][:200]}...\n")
