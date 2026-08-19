"""
Improved retriever with expanded knowledge base for any question
Generates knowledge base from same sources as general_qa_data
"""

# Expanded knowledge base - many more facts for any question
KNOWLEDGE_BASE_V2 = [
    # Geography - individual countries
    "France: Capital is Paris. Country in Europe. Known for Eiffel Tower.",
    "Germany: Capital Berlin. In Europe. Known for engineering.",
    "Japan: Capital Tokyo. In East Asia. Known for technology and sushi.",
    "USA: Capital Washington D.C. In North America. Largest economy.",
    "UK: Capital London. In Europe. Known for Big Ben.",
    "Canada: Capital Ottawa. In North America. Known for maple syrup.",
    "Australia: Capital Canberra. In Oceania. Known for kangaroos.",
    "Brazil: Capital Brasilia. In South America. Known for Amazon.",
    "India: Capital New Delhi. In Asia. Population over 1 billion.",
    "China: Capital Beijing. In Asia. Most populous country.",
    "Russia: Capital Moscow. Spans Europe and Asia. Largest country by area.",
    "Italy: Capital Rome. Known for pizza and Colosseum.",
    "Spain: Capital Madrid. Known for flamenco.",
    "Mexico: Capital Mexico City. In North America.",
    "Egypt: Capital Cairo. Known for pyramids.",
    "South Korea: Capital Seoul. Known for K-pop.",
    "Sweden: Capital Stockholm. Known for IKEA.",
    "Norway: Capital Oslo. Known for fjords.",
    "Switzerland: Capital Bern. Known for watches and chocolate.",
    "New Zealand: Capital Wellington. Known for kiwi bird.",
    # Elements detailed
    "Elements: Hydrogen H atomic number 1 lightest element. Helium He number 2 used in balloons. Carbon C number 6 basis of life. Oxygen O number 8 we breathe. Gold Au number 79 precious metal. Silver Ag 47. Iron Fe 26 in blood. Sodium Na 11 in salt.",
    "Chemical symbols: H Hydrogen, He Helium, Li Lithium, Be Beryllium, B Boron, C Carbon, N Nitrogen, O Oxygen, F Fluorine, Ne Neon, Na Sodium, Mg Magnesium, Al Aluminum, Si Silicon, P Phosphorus, S Sulfur, Cl Chlorine, K Potassium, Ca Calcium, Fe Iron, Cu Copper, Zn Zinc, Ag Silver, Au Gold, Hg Mercury, Pb Lead.",
    # Planets detailed
    "Solar system: 8 planets. Mercury smallest closest to sun. Venus hottest 462C thick atmosphere. Earth only life 71% water. Mars red planet has Olympus Mons largest volcano. Jupiter largest gas giant Great Red Spot. Saturn has rings made of ice. Uranus ice giant tilted 98 degrees. Neptune farthest strong winds.",
    "Earth: Diameter 12,742 km. Orbits sun in 365 days. Moon is its satellite. 71% water 29% land. Atmosphere has oxygen.",
    "Sun: Star at center solar system. Provides light and heat. Made of hydrogen and helium. Temperature 5,500C surface.",
    # Human body
    "Human body: 206 bones adult. Largest organ skin. Heart pumps blood. Brain controls body. Lungs breathe oxygen. Stomach digests food. 5 senses: sight hearing smell taste touch.",
    "Blood: Red blood cells carry oxygen. White blood cells fight infection. Plasma is liquid part.",
    # Physics
    "Physics: Speed of light 299,792 km/s. Gravity pulls objects. Newton's laws: 1st object at rest stays, 2nd F=ma, 3rd action reaction. Energy cannot be created destroyed.",
    "Water: Formula H2O. Boils at 100C freezes at 0C. Three states solid ice liquid water gas steam.",
    # Biology
    "Photosynthesis: Plants use sunlight + CO2 + water to make food and oxygen. Occurs in leaves chlorophyll. Equation 6CO2+6H2O+light -> C6H12O6+6O2.",
    "Animals: Dogs have 4 legs friendly. Cats have 9 lives myth. Birds have wings fly. Fish live in water have gills. Bees make honey pollinate.",
    # History
    "History: World War 1 1914-1918. World War 2 1939-1945 ended 1945. Moon landing 1969 Apollo 11 Neil Armstrong. Light bulb Thomas Edison 1879. Berlin Wall fell 1989. Mona Lisa Leonardo da Vinci.",
    "American history: USA independence 1776. First president George Washington. Civil war 1861-1865. Martin Luther King civil rights.",
    # Math
    "Math basics: Addition +, subtraction -, multiplication *, division /. Percentage % is per hundred. 10% of 100 is 10. Average sum divided by count.",
    "Geometry: Triangle 3 sides, square 4 equal sides, circle round 360 degrees, rectangle 4 sides opposite equal.",
    # Definitions
    "Definitions: Computer electronic device processes data. Algorithm step-by-step instructions like recipe. AI Artificial Intelligence computers thinking like humans. Internet global network. Gravity force pulling objects. Photosynthesis plant food making. Ecosystem community living things.",
    "What is time? Time measures duration. What is energy? Energy ability to do work. What is matter? Matter anything with mass.",
    # How-to
    "How to: Paper airplane fold paper half lengthwise unfold fold corners center fold again make wings. Boil egg put in water boil 10 min cool peel. Save money track spending budget avoid unnecessary buys. Learn programming pick language like Python practice daily build projects.",
    "Cooking: To make pasta boil water add pasta 8-10 min drain. To make sandwich put fillings between bread.",
    # Commonsense
    "Commonsense: Wear jacket winter to stay warm. Birds fly south winter for warmth and food. Drink water to stay hydrated. Day night because Earth rotates. Use pen to write. Cars need fuel to move. Plants need sunlight water.",
    "Seasons: Spring flowers bloom, Summer hot, Fall leaves change, Winter cold snow. Caused by Earth tilt.",
    # Open ended / philosophy
    "Meaning of life: Many think meaning is happiness helping others learning making world better. Different cultures have different answers.",
    "How to be happy: Spend time loved ones, do enjoyable things, help others, exercise sleep, gratitude.",
    "Study tips: Make plan, short focused sessions, take notes, test yourself, rest well.",
    # Tech
    "Python: Programming language easy to learn. Created by Guido van Rossum. Used for AI web data. Print hello: print(\"Hello\"). Function def add(a,b): return a+b.",
    "Internet: Websites have domain like .com. Browser like Chrome shows websites. Email sends messages.",
]

# Also include all original knowledge base
try:
    from .retriever import KNOWLEDGE_BASE as OLD_KB
    FULL_KB = OLD_KB + KNOWLEDGE_BASE_V2
except:
    FULL_KB = KNOWLEDGE_BASE_V2

def retrieve_v2(query, top_k=3):
    """Improved retrieval with better scoring"""
    import re
    query_lower = query.lower()
    keywords = re.findall(r'\b\w{3,}\b', query_lower)
    if not keywords:
        return []

    scored = []
    for chunk in FULL_KB:
        chunk_lower = chunk.lower()
        score = 0
        for kw in keywords:
            if kw in chunk_lower:
                # weight longer keywords more
                score += len(kw) * 0.1 + 1
        # bonus for exact phrase match
        if query_lower in chunk_lower:
            score += 5
        if score>0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s,c in scored[:top_k] if s>0]

if __name__ == "__main__":
    tests = ["What is capital of Japan?", "Gold symbol?", "How many planets?", "What is photosynthesis?", "How to make paper airplane?", "What is AI?", "When did WW2 end?"]
    for q in tests:
        print(f"Q: {q}")
        print("Retrieved:", retrieve_v2(q, top_k=2))
        print()
