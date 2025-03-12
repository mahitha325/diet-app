import os
import logging
import streamlit as st
from pathlib import Path
from datetime import datetime
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import nltk

# Set the NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# ========== INITIALIZATION ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

 # ========== HEALTH KNOWLEDGE BASE ==========
def initialize_health_index():
    try:
        index_path = Path("health_index")
        index_path.mkdir(exist_ok=True)
        
        if (index_path / "docstore.json").exists():
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            return VectorStoreIndex.from_documents(
                documents=[],
                storage_context=storage_context
            )
        
        # Create health knowledge base with improved structure
        documents = [
            Document(text="""Condition: Hypertension | High Blood Pressure
                Diagnostic Criteria: BP > 140/90 mmHg
                Recommendations:
                - Sodium restriction <1500mg/day
                - Potassium-rich foods: bananas, spinach, sweet potatoes
                - Whole grains and lean proteins
                - Limit alcohol/caffeine"""),
            
            Document(text="""Condition: Diabetes | High Blood Sugar
                Diagnostic Criteria: Fasting glucose > 126 mg/dL
                Recommendations:
                - Low glycemic index foods
                - Balanced carbohydrate distribution
                - High fiber intake
                - Healthy fats: avocado, nuts"""),
                
            Document(text="""Condition: Hyperlipidemia | High Cholesterol
                Diagnostic Criteria: LDL > 130 mg/dL
                Recommendations:
                - Reduce saturated fats
                - Omega-3 sources: fish, flaxseeds
                - Soluble fiber: oats, beans
                - Plant sterols/stanols"""),
                
            Document(text="""Condition: Weight Management
                Diagnostic Criteria: BMI > 25
                Recommendations:
                - Calorie deficit: 500-750 kcal/day
                - High protein intake
                - Portion control
                - Regular exercise""")
        ]
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=index_path)
        return index
    except Exception as e:
        logger.error(f"Index initialization error: {str(e)}")
        st.error("Failed to load health knowledge base")
        return None

health_index = initialize_health_index()

# ========== HEALTH CALCULATIONS ==========
def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

def analyze_health(age, weight, height, bp, sugar, cholesterol):
    results = {
        "bmi": round(calculate_bmi(weight, height), 1),
        "conditions": [],
        "age": age
    }
    
    # BMI Classification
    if results["bmi"] < 18.5:
        results["conditions"].append("Underweight")
    elif 18.5 <= results["bmi"] < 25:
        results["conditions"].append("Normal weight")
    elif 25 <= results["bmi"] < 30:
        results["conditions"].append("Overweight")
    else:
        results["conditions"].append("Obese")
    
    # Blood Pressure Analysis
    try:
        systolic, diastolic = map(int, bp.split('/'))
        if systolic >= 140 or diastolic >= 90:
            results["conditions"].append("Hypertension")
    except:
        st.error("Invalid blood pressure format. Use systolic/diastolic (e.g., 120/80)")
    
 
    # Blood Sugar Analysis
    if sugar >= 126:
        results["conditions"].append("High Blood Sugar")
    
    # Cholesterol Analysis
    if cholesterol >= 200:
        results["conditions"].append("High Cholesterol")
    
    return results

# ========== DIET GENERATION ==========
def generate_diet_plan(health_data, context):
    try:
        # Base recommendations
        recommendations = [
            "Balanced nutrition with variety of foods",
            "Stay hydrated (8 glasses of water daily)",
            "Regular meal timings"
        ]
        
        # Process context from knowledge base
        if context:
            try:
                context_recommendations = []
                for line in context.split('\n'):
                    if line.strip().startswith("- "):
                        context_recommendations.append(line.strip())
                if context_recommendations:
                    recommendations = context_recommendations
            except Exception as e:
                logger.error(f"Context processing error: {str(e)}")
        
        # Create meal plan template
        meal_plan = {
            "Breakfast": "Whole grain cereal with fruits and nuts",
            "Morning Snack": "Greek yogurt or fresh fruit",
            "Lunch": "Grilled protein with vegetables and quinoa",
            "Afternoon Snack": "Vegetable sticks with hummus",
            "Dinner": "High-fiber meal with lean protein and salad"
        }
        
        # Format output
        return f"""
## Personalized Diet Plan for {health_data['age']} Year Old

### Health Summary:
- **BMI:** {health_data['bmi']} ({health_data['conditions'][0]})
- **Identified Conditions:** {', '.join(health_data['conditions'])}

### Dietary Recommendations:
{'\n'.join([f'- {rec}' for rec in recommendations])}

### Sample Daily Meal Plan:
{'\n'.join([f'**{meal}:** {details}' for meal, details in meal_plan.items()])}

### Lifestyle Advice:
- Engage in 30 minutes of moderate exercise daily
- Practice stress-reduction techniques
- Get 7-8 hours of quality sleep
- Regular health checkups

*Based on analysis of your health metrics and medical guidelines*
"""
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return None

# ========== MAIN INTERFACE ==========
def main():
    st.title("AI-Powered Health Advisor 🩺")
    
    # Input Section
    with st.expander("Enter Your Health Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            weight = st.number_input("Weight (kg)", min_value=30, value=70)
            height = st.number_input("Height (cm)", min_value=100, value=170)
        with col2:
            bp = st.text_input("Blood Pressure (e.g., 120/80)", value="120/80")
            sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, value=100)
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, value=180)
    
    if st.button("Generate Personalized Diet Plan", type="primary"):
        if not health_index:
            st.error("Health knowledge base not loaded. Please restart the app.")
            return
            
        with st.spinner("Analyzing your health profile and generating recommendations..."):
            try:
                # Health analysis
                health_data = analyze_health(age, weight, height, bp, sugar, cholesterol)
                
                # Retrieve medical context
                retriever = health_index.as_retriever(similarity_top_k=2)
                query = " ".join(health_data['conditions'])
                retrieved_nodes = retriever.retrieve(query)
                context = "\n".join([n.node.text for n in retrieved_nodes])
                
                # Generate diet plan
                diet_plan = generate_diet_plan(health_data, context)
                
                if diet_plan:
                    st.success("✅ Your personalized plan is ready!")
                    st.markdown(diet_plan)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Full Plan",
                        data=diet_plan,
                        file_name=f"health_plan_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate plan. Please try again.")
                    
            except Exception as e:
                logger.error(f"System error: {str(e)}")
                st.error("Failed to generate plan. Please check your inputs and try again.")

    # Disclaimer
    st.markdown("""
    ---
    **Disclaimer:** This AI system provides general health information and should not be used as a substitute for professional medical advice. Always consult a qualified healthcare provider before making any changes to your diet or lifestyle.
    """)

if __name__ == "__main__":
    main()