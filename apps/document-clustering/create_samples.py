"""
Create sample documents for clustering demo
"""
from pathlib import Path

# Create sample_documents directory
sample_dir = Path(__file__).parent / "sample_documents"
sample_dir.mkdir(exist_ok=True)

# Sample documents - diverse topics for clear clustering
documents = {
    # CATEGORY: Technology/AI (5 docs)
    "ai_machine_learning.txt": """
Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Neural networks, inspired by the human brain, form the foundation of deep learning. These algorithms can recognize patterns, make predictions, and improve their performance over time through training on large datasets. Applications include computer vision, natural language processing, and autonomous systems.
""",

    "quantum_computing.txt": """
Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to perform calculations. Unlike classical bits that are either 0 or 1, quantum bits or qubits can exist in multiple states simultaneously. This enables quantum computers to solve certain problems exponentially faster than classical computers. IBM, Google, and other companies are developing quantum processors with increasing qubit counts.
""",

    "blockchain_technology.txt": """
Blockchain is a distributed ledger technology that maintains a continuously growing list of records called blocks. Each block contains a cryptographic hash of the previous block, creating an immutable chain. This technology enables secure, transparent, and decentralized transactions without intermediaries. Cryptocurrencies like Bitcoin and Ethereum are built on blockchain infrastructure.
""",

    "artificial_intelligence.txt": """
Artificial intelligence refers to machines that can perform tasks requiring human intelligence. AI systems use algorithms to process information, learn from experience, and make decisions. Modern AI includes machine learning, computer vision, robotics, and natural language understanding. Recent advances in neural networks have led to breakthroughs in image recognition, language translation, and game playing.
""",

    "cybersecurity.txt": """
Cybersecurity protects computer systems, networks, and data from digital attacks and unauthorized access. Security measures include encryption, firewalls, intrusion detection systems, and authentication protocols. Common threats include malware, phishing, ransomware, and data breaches. Organizations employ security professionals and implement multi-layered defense strategies to safeguard sensitive information.
""",

    # CATEGORY: Health/Medicine (5 docs)
    "cardiovascular_health.txt": """
The cardiovascular system includes the heart, blood vessels, and blood that circulate throughout the body. Regular exercise strengthens the heart muscle and improves circulation. A healthy diet low in saturated fats and high in fruits and vegetables reduces risk of heart disease. Monitoring blood pressure and cholesterol levels helps prevent cardiovascular problems.
""",

    "nutrition_wellness.txt": """
Proper nutrition provides essential nutrients for optimal health and wellbeing. A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals. Whole foods like vegetables, fruits, whole grains, and lean proteins support immune function and energy levels. Staying hydrated, limiting processed foods, and eating mindfully contribute to overall wellness.
""",

    "mental_health.txt": """
Mental health encompasses emotional, psychological, and social wellbeing. It affects how we think, feel, and behave in daily life. Common mental health conditions include anxiety, depression, and stress-related disorders. Treatment approaches include therapy, medication, mindfulness practices, and lifestyle changes. Seeking professional help and maintaining social connections are important for mental wellness.
""",

    "immunology_vaccines.txt": """
The immune system defends the body against infections and diseases. White blood cells, antibodies, and other mechanisms identify and destroy pathogens. Vaccines work by training the immune system to recognize specific diseases without causing illness. Immunization has eliminated or greatly reduced many infectious diseases like polio, measles, and smallpox worldwide.
""",

    "diabetes_management.txt": """
Diabetes is a chronic condition affecting how the body processes blood sugar. Type 1 diabetes results from insufficient insulin production, while Type 2 involves insulin resistance. Managing diabetes requires monitoring blood glucose levels, taking medications as prescribed, eating healthy foods, and exercising regularly. Untreated diabetes can lead to serious complications affecting eyes, kidneys, nerves, and cardiovascular system.
""",

    # CATEGORY: Environment/Climate (5 docs)
    "climate_change.txt": """
Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly burning fossil fuels, release greenhouse gases that trap heat in the atmosphere. Rising temperatures cause melting ice caps, sea level rise, and extreme weather events. Reducing carbon emissions and transitioning to renewable energy are crucial for mitigating climate impacts.
""",

    "renewable_energy.txt": """
Renewable energy comes from natural sources that replenish continuously. Solar panels convert sunlight into electricity, while wind turbines harness wind power. Hydroelectric dams generate power from flowing water. These clean energy sources produce minimal greenhouse gas emissions compared to fossil fuels. Battery storage technology enables renewable energy to provide reliable baseload power.
""",

    "ocean_conservation.txt": """
Oceans cover over 70 percent of Earth's surface and support diverse marine ecosystems. Ocean pollution from plastics, chemicals, and oil spills threatens marine life and coral reefs. Overfishing depletes fish populations and disrupts food chains. Conservation efforts include marine protected areas, sustainable fishing practices, and reducing plastic waste. Healthy oceans regulate climate and provide food for billions of people.
""",

    "biodiversity_ecosystems.txt": """
Biodiversity refers to the variety of life on Earth, including different species, genes, and ecosystems. Tropical rainforests, coral reefs, and wetlands harbor rich biodiversity. Habitat destruction, pollution, and climate change threaten species with extinction. Protecting biodiversity maintains ecosystem services like pollination, water purification, and climate regulation essential for human survival.
""",

    "sustainable_agriculture.txt": """
Sustainable agriculture produces food while preserving environmental resources for future generations. Practices include crop rotation, composting, water conservation, and integrated pest management. Organic farming avoids synthetic pesticides and fertilizers. Sustainable methods improve soil health, reduce pollution, and support local food systems. Regenerative agriculture can even sequester carbon and help mitigate climate change.
""",

    # CATEGORY: Sports/Fitness (4 docs)
    "sports_training.txt": """
Athletic training develops strength, speed, endurance, and skill for sports performance. Training programs include cardiovascular exercise, resistance training, flexibility work, and sport-specific drills. Proper warm-up prevents injuries, while cool-down aids recovery. Nutrition and rest are essential components of effective training. Athletes track progress and adjust training based on performance goals.
""",

    "marathon_running.txt": """
Marathon running is a long-distance race covering 26.2 miles. Training for a marathon requires gradually building weekly mileage over several months. Long runs develop endurance, while tempo runs and intervals improve speed. Proper running shoes, hydration during runs, and recovery days prevent overuse injuries. Race day strategy includes pacing, fueling, and mental preparation.
""",

    "strength_conditioning.txt": """
Strength and conditioning programs build muscle, power, and athletic performance. Compound exercises like squats, deadlifts, and bench presses work multiple muscle groups. Progressive overload gradually increases weight or resistance over time. Adequate protein intake supports muscle growth and recovery. Periodization cycles training intensity and volume to optimize results and prevent plateaus.
""",

    "yoga_flexibility.txt": """
Yoga combines physical postures, breathing techniques, and meditation for mind-body wellness. Regular practice improves flexibility, balance, and core strength. Different styles include vigorous vinyasa flow and gentle restorative yoga. Yoga reduces stress, enhances body awareness, and promotes relaxation. Breathing exercises calm the nervous system and improve focus.
""",
}

# Write all documents
for filename, content in documents.items():
    filepath = sample_dir / filename
    filepath.write_text(content.strip())
    print(f"Created: {filename}")

print(f"\n✓ Created {len(documents)} sample documents in {sample_dir}")
print("\nCategories:")
print("  - Technology/AI: 5 documents")
print("  - Health/Medicine: 5 documents")
print("  - Environment/Climate: 5 documents")
print("  - Sports/Fitness: 4 documents")
