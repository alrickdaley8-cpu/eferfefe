"""
Simple retrieval for TinyLLM to answer any question
Uses in-memory knowledge base from general_qa_data to augment prompt
Makes 1M model appear to know any fact via RAG
"""
import re

# Knowledge base for retrieval - same as general_qa_data but expanded as text chunks
KNOWLEDGE_BASE = [
    # Geography
    "France's capital is Paris. France is in Europe.",
    "Germany's capital is Berlin. Germany is in Europe.",
    "Japan's capital is Tokyo. Japan is in East Asia.",
    "USA's capital is Washington D.C. USA is in North America.",
    "UK's capital is London. UK is in Europe.",
    "Canada's capital is Ottawa. Canada is in North America.",
    "Australia's capital is Canberra. Australia is in Oceania.",
    "Brazil's capital is Brasilia. Brazil is in South America.",
    "India's capital is New Delhi. India is in Asia.",
    "China's capital is Beijing. China is in Asia.",
    "Russia's capital is Moscow. Russia is in Europe and Asia.",
    "Italy capital Rome, Spain capital Madrid, Mexico capital Mexico City.",
    "Egypt capital Cairo, South Korea capital Seoul, Australia Canberra.",
    # Elements
    "Chemical symbols: Hydrogen H, Helium He, Lithium Li, Carbon C, Nitrogen N, Oxygen O, Sodium Na, Magnesium Mg, Aluminum Al, Silicon Si, Iron Fe, Copper Cu, Silver Ag, Gold Au, Mercury Hg, Lead Pb, Uranium U.",
    "Gold symbol Au atomic number 79, Silver Ag 47, Iron Fe 26, Copper Cu 29, Oxygen O 8, Hydrogen H 1.",
    # Planets
    "Solar system has 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Mercury closest to sun, Venus hottest, Earth only with life, Mars red planet, Jupiter largest, Saturn has rings, Uranus tilted, Neptune farthest.",
    # Science
    "Largest organ in human body is skin. Adult human has 206 bones. Speed of light is 299,792 km/s. Water formula H2O. Humans breathe oxygen O2, plants take CO2. Photosynthesis is how plants make food from sunlight.",
    "Earth rotates causing day and night. Sun is a star. Moon orbits Earth. Gravity pulls objects together.",
    # History
    "World War 2 ended in 1945. Moon landing 1969 Apollo 11. Light bulb invented by Thomas Edison 1879. Berlin Wall fell 1989. Mona Lisa painted by Leonardo da Vinci.",
    # Math help
    "Math: addition adds numbers, subtraction takes away, multiplication is repeated addition, percentage is part of 100.",
    # Definitions
    "Computer is electronic device processing data. Algorithm is step-by-step instructions. AI is Artificial Intelligence computers thinking like humans. Internet is global network connecting computers. Gravity is force pulling objects together.",
    # How to
    "Paper airplane: fold paper half lengthwise, fold corners to center, fold again, make wings. Boil egg: put in water, boil 10 min, cool, peel.",
]

def retrieve(query, top_k=3):
    """Simple keyword retrieval - returns relevant knowledge chunks"""
    query_lower = query.lower()
    # Extract keywords (words >2 chars)
    keywords = re.findall(r'\b\w{3,}\b', query_lower)
    if not keywords:
        return []

    scored = []
    for chunk in KNOWLEDGE_BASE:
        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            if kw in chunk_lower:
                score += 1
        # bonus for exact country/element names
        if score>0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s,c in scored[:top_k] if s>0]

def build_context_prompt(retrieved):
    if not retrieved:
        return ""
    return "Relevant knowledge:\n" + "\n".join(f"- {r}" for r in retrieved) + "\n"

if __name__ == "__main__":
    tests = ["What is capital of France?", "What is gold symbol?", "How many planets?", "What is photosynthesis?"]
    for q in tests:
        print(f"Q: {q}")
        print(f"Retrieved: {retrieve(q)}")
        print()
