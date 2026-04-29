# utils/ai_assistant.py
"""
AI Assistant Module for Heart Attack Risk Prediction
Uses Google Gemini API (FREE) with automatic fallback
"""

import os
import streamlit as st

class HeartifyAIAssistant:
    """AI Assistant for cardiovascular health questions using Gemini"""
    
    def __init__(self):
        """Initialize the AI assistant with Gemini API"""
        self.api_key = None
        self.is_available = False
        self.error_message = None
        self.client = None
        self.model_name = "gemini-2.0-flash"
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini with API key from secrets or environment"""
        
        # Step 1: Try to get API key
        try:
            # Try from Streamlit secrets
            try:
                self.api_key = st.secrets.get("GEMINI_API_KEY", None)
                if self.api_key:
                    print(f"Found GEMINI_API_KEY in secrets")
            except:
                pass
            
            # Try from environment variable
            if not self.api_key:
                self.api_key = os.environ.get("GEMINI_API_KEY", None)
                if self.api_key:
                    print(f"Found GEMINI_API_KEY in environment")
                    
        except Exception as e:
            print(f"Error getting API key: {e}")
        
        # Step 2: If no API key, use fallback mode
        if not self.api_key:
            self.is_available = False
            self.error_message = "⚠️ Using offline mode. Add GEMINI_API_KEY to .streamlit/secrets.toml for AI features."
            return
        
        # Step 3: Try to initialize Gemini
        try:
            # Import the new Google GenAI SDK
            import google.genai as genai
            
            # Initialize client
            self.client = genai.Client(api_key=self.api_key)
            
            # Test the connection with a simple request
            test_response = self.client.models.generate_content(
                model=self.model_name,
                contents="Test"
            )
            
            self.is_available = True
            self.error_message = None
            print(f"✅ Gemini AI initialized successfully with model: {self.model_name}")
            
        except ImportError:
            self.is_available = False
            self.error_message = "⚠️ Google GenAI package not installed. Run: pip install google-genai"
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                self.is_available = False
                self.error_message = "⚠️ API quota exceeded. Using offline mode. Get a new key from https://makersuite.google.com/app/apikey"
            elif "403" in error_str or "permission" in error_str.lower():
                self.is_available = False
                self.error_message = "⚠️ Invalid API key. Please check your GEMINI_API_KEY in .streamlit/secrets.toml"
            else:
                self.is_available = False
                self.error_message = f"⚠️ Using offline mode. Error: {error_str[:100]}"
    
    def get_response(self, question, patient_context=None, model_type=None):
        """
        Get response from Gemini API or fallback to offline mode
        
        Args:
            question (str): User's question
            patient_context (dict, optional): Current patient data for context
            model_type (str): Not used (kept for compatibility)
        
        Returns:
            str: AI response
        """
        # If API is not available, use offline responses
        if not self.is_available:
            return self._get_offline_response(question)
        
        # System prompt for medical context
        system_prompt = """You are Heartify AI, a knowledgeable and empathetic cardiovascular health assistant. 
        You provide evidence-based information about heart attack risk factors, prevention, lifestyle changes, 
        and general cardiovascular health. Always include a disclaimer that you're not a substitute for professional 
        medical advice. Keep responses concise (2-3 paragraphs max) and helpful.
        
        Guidelines:
        - Be accurate and cite general medical knowledge
        - Encourage healthy lifestyle choices
        - Never diagnose or prescribe treatment
        - Always recommend consulting healthcare providers
        - Be sensitive and supportive
        - Use bullet points for lists when helpful"""
        
        # Add patient context if available
        context_text = ""
        if patient_context:
            context_text = f"""
            
[EDUCATIONAL CONTEXT ONLY - Patient data for reference]
Age: {patient_context.get('age')}, Gender: {patient_context.get('gender')}
BMI: {patient_context.get('bmi')}, BP: {patient_context.get('systolic_bp')}/{patient_context.get('diastolic_bp')}
Cholesterol: LDL {patient_context.get('ldl')}, HDL {patient_context.get('hdl')}
Provide only general educational information. Do NOT give specific medical advice."""
        
        full_prompt = f"{system_prompt}\n\nUser question: {question}{context_text}"
        
        try:
            # Try to use Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            return response.text
            
        except Exception as e:
            # If API fails, fallback to offline responses
            return self._get_offline_response(question)
    
    def _get_offline_response(self, question):
        """Comprehensive fallback responses when API is not available"""
        q = question.lower()
        
        # Blood pressure and cholesterol combined
        if "blood pressure" in q and "cholesterol" in q:
            return """**🩸 Ideal Blood Pressure & Cholesterol Levels**

**Blood Pressure (mmHg):**
- Normal: <120/80
- Elevated: 120-129/<80  
- High: ≥130/80

**Cholesterol (mg/dL):**
- Total: <200 (desirable)
- LDL (Bad): <100 (optimal)
- HDL (Good): >40 (men), >50 (women)
- Triglycerides: <150

**How to Improve:**
- Eat Mediterranean or DASH diet
- Exercise 150+ minutes/week
- Limit sodium and saturated fats
- Take medications as prescribed

*Note: General guidelines only. Consult your doctor for personalized targets.*"""
        
        # Blood pressure only
        elif "blood pressure" in q or "hypertension" in q:
            return """**🩸 Understanding Blood Pressure**

**Ideal Levels (mmHg):**
- Normal: <120/80
- Elevated: 120-129/<80
- Stage 1 Hypertension: 130-139/80-89
- Stage 2 Hypertension: ≥140/90

**Management Tips:**
- Reduce sodium (<2300mg/day)
- Exercise 30 minutes daily
- Limit alcohol (1-2 drinks/day)
- Maintain healthy weight (BMI 18.5-24.9)
- Take medications as prescribed

**Why It Matters:**
High blood pressure damages arteries and increases heart attack risk by 2-3x.

*Monitor at home and keep a log for your doctor.*"""
        
        # Cholesterol only
        elif "cholesterol" in q or "ldl" in q or "hdl" in q:
            return """**📊 Understanding Cholesterol Levels**

**Target Numbers (mg/dL):**

**LDL (Bad Cholesterol):**
- Optimal: <100
- Near optimal: 100-129
- Borderline high: 130-159
- High: 160-189

**HDL (Good Cholesterol):**
- Poor: <40 (men), <50 (women)
- Better: 40-59
- Best: ≥60

**Total Cholesterol:**
- Desirable: <200
- Borderline: 200-239
- High: ≥240

**Triglycerides:**
- Normal: <150
- Borderline: 150-199
- High: 200-499

**How to Improve Naturally:**
- Eat oats, nuts, fatty fish, olive oil
- Exercise regularly (150+ min/week)
- Reduce saturated and trans fats
- Increase soluble fiber (beans, apples)

*Note: Some people need statins regardless of lifestyle.*"""
        
        # Diet questions
        elif "diet" in q or "mediterranean" in q or "dash" in q or "eat" in q:
            return """**🥗 Heart-Healthy Eating Guide**

**Mediterranean Diet Basics:**
- **Daily**: Vegetables, fruits, whole grains, nuts, olive oil
- **Weekly**: Fish, poultry, beans, eggs
- **Occasionally**: Red meat, sweets

**Foods to Eat More:**
- 🐟 Omega-3 fish (salmon, mackerel, sardines)
- 🫐 Berries and leafy greens
- 🌾 Oats, quinoa, brown rice
- 🥜 Nuts and seeds (walnuts, chia, flax)
- 🥑 Avocados and olive oil

**Foods to Limit:**
- Processed meats (bacon, sausage)
- Fried foods and fast food
- Sugary drinks and sweets
- Refined carbs (white bread, pastries)
- Excess sodium

**Sample Heart-Healthy Day:**
- Breakfast: Oatmeal with berries and walnuts
- Lunch: Quinoa salad with chickpeas and avocado
- Dinner: Grilled salmon with roasted vegetables
- Snack: Apple with almond butter

*Start with small changes - add one vegetable serving daily!*"""
        
        # Exercise questions
        elif "exercise" in q or "physical activity" in q or "workout" in q:
            return """**🏃 Exercise for Heart Health**

**Recommended Amount:**
- **150 minutes/week** moderate activity OR
- **75 minutes/week** vigorous activity
- **Strength training**: 2+ days/week

**What Counts?**
- **Moderate**: Brisk walking, light jogging, cycling, swimming, dancing
- **Vigorous**: Running, fast cycling, HIIT, competitive sports

**Sample Weekly Plan:**
- Monday: 30 min brisk walking
- Tuesday: 30 min strength training
- Wednesday: 45 min cycling
- Thursday: 30 min brisk walking
- Friday: 30 min swimming
- Saturday: 60 min hiking
- Sunday: Rest or gentle yoga

**Benefits:**
- Reduces heart attack risk by 20-35%
- Lowers blood pressure and cholesterol
- Helps maintain healthy weight
- Reduces stress and improves sleep

*Start with 10-15 minute sessions if new to exercise. Consistency matters more than intensity!*"""
        
        # Risk factors
        elif "risk factor" in q:
            return """**⚠️ Top Heart Attack Risk Factors**

**Modifiable (You Can Change):**
1. **High Blood Pressure** (>130/80 mmHg)
2. **High Cholesterol** (LDL >100 mg/dL)
3. **Smoking** (doubles heart attack risk)
4. **Diabetes** (2-3x higher risk)
5. **Obesity** (BMI ≥30)
6. **Physical Inactivity**
7. **Poor Diet** (high in saturated fats, sugar)
8. **Stress & Poor Sleep**
9. **Excessive Alcohol**

**Non-Modifiable (Can't Change):**
- Age (>45 for men, >55 for women)
- Family history of heart disease
- Gender (men at higher risk earlier)
- Ethnicity

**Risk Level by Number of Factors:**
- 0-1 factors: Low risk
- 2-3 factors: Moderate risk
- 4+ factors: High risk

*Most heart attacks are preventable! Focus on changing modifiable factors first.*"""
        
        # Warning signs
        elif "warning sign" in q or "symptom" in q or "heart attack sign" in q:
            return """**⚠️ Heart Attack Warning Signs**

**Main Symptoms:**
- **Chest discomfort**: Pressure, squeezing, fullness, or pain in center/left chest
- **Upper body pain**: Spreading to shoulders, arms (especially left), back, neck, jaw
- **Shortness of breath**: With or without chest discomfort
- **Other signs**: Cold sweat, nausea, lightheadedness, unusual fatigue

**⚠️ Women Often Experience Different Symptoms:**
- Unusual fatigue (weeks before a heart attack)
- Sleep disturbances
- Shortness of breath
- Indigestion or anxiety
- Back or jaw pain

**🚨 WHAT TO DO:**
1. **CALL 911 IMMEDIATELY** (don't drive yourself)
2. Chew aspirin if not allergic
3. Stay calm and rest
4. Unlock door for emergency responders
5. Tell operator "possible heart attack"

**Don't wait more than 5 minutes - every minute matters!**

*Some people have no symptoms ("silent heart attack"). Regular check-ups are essential for prevention.*"""
        
        # Prevention
        elif "prevent" in q or "prevention" in q:
            return """**💪 7 Evidence-Based Prevention Strategies**

**1. Know Your Numbers**
- Blood pressure (<120/80 mmHg)
- Cholesterol (LDL <100, HDL >40)
- Blood sugar (fasting <100 mg/dL)
- BMI (18.5-24.9)

**2. Eat Heart-Healthy Diet**
- Mediterranean or DASH diet
- More fruits, vegetables, whole grains
- Limit saturated fats, sugar, sodium

**3. Exercise Regularly**
- 150+ minutes moderate activity/week
- Include strength training 2x/week

**4. Don't Smoke**
- Single best thing for your heart
- Benefits start immediately after quitting

**5. Manage Stress**
- Meditation, deep breathing, yoga
- 7-9 hours quality sleep
- Maintain social connections

**6. Limit Alcohol**
- Men: max 2 drinks/day
- Women: max 1 drink/day

**7. Take Medications as Prescribed**
- Never stop heart medications without consulting doctor

*Start with ONE change today. Small improvements add up to big results!*"""
        
        # Smoking
        elif "smoking" in q or "smoke" in q:
            return """**🚭 Smoking & Heart Disease**

**How Smoking Damages Your Heart:**
- Reduces oxygen in blood
- Damages artery lining (accelerates atherosclerosis)
- Increases blood pressure and heart rate
- Makes blood stickier (increases clot risk)
- Reduces HDL (good cholesterol)

**Recovery Timeline After Quitting:**
- **20 minutes**: BP and heart rate drop
- **12 hours**: CO levels normalize
- **2 weeks-3 months**: Circulation improves
- **1 year**: Heart disease risk drops by 50%
- **15 years**: Heart disease risk equals non-smoker

**Quitting Strategies:**
- Nicotine replacement (patches, gum, lozenges)
- Prescription medications (Chantix, Zyban)
- Counseling or support groups
- Quit apps (Smoke Free, QuitNow)
- Call 1-800-QUIT-NOW for free coaching

*Every attempt improves success - keep trying! It's never too late to quit.*"""
        
        # Family history
        elif "family history" in q or "genetic" in q or "hereditary" in q:
            return """**🧬 Family History & Heart Disease**

**Understanding Your Risk:**
Having a first-degree relative (parent, sibling) with heart disease:
- Before 55 (men) or 65 (women) increases your risk 2-3x
- Two or more relatives increases risk 4-5x

**What You Can Do:**

**1. Get Comprehensive Screening:**
- Start at age 20-30 (earlier than general population)
- Get Lipoprotein(a) test (genetic cholesterol marker)

**2. Aggressive Lifestyle Measures:**
- Maintain ideal BMI (18.5-24.9)
- Never smoke
- Exercise 300+ minutes/week
- Strict heart-healthy diet

**3. Medical Management:**
- May need statins at lower thresholds
- Earlier blood pressure treatment

**4. Regular Monitoring:**
- Yearly physicals with lipid panel
- Track all numbers

*Important: Family history is NOT destiny. You can significantly reduce your risk with proper management!*"""
        
        # Default response
        else:
            return """**💙 Heart Health Assistant**

I'm here to help you understand cardiovascular health. I can provide information about:

**Topics I Cover:**
- ❤️ **Risk factors** - What increases heart attack risk
- 🛡️ **Prevention** - Lifestyle changes to protect your heart
- 🩺 **Medical concepts** - Blood pressure, cholesterol, diabetes
- 🍎 **Diet & nutrition** - Heart-healthy eating (Mediterranean, DASH)
- 🏃 **Exercise** - Activity recommendations
- 🧘 **Stress management** - Protecting mental health
- ⚠️ **Symptoms** - Warning signs of heart attack
- 📊 **Test results** - Understanding your numbers

**Try asking specific questions:**
- "What are ideal blood pressure levels?"
- "How can I lower my cholesterol naturally?"
- "What's a heart-healthy diet?"
- "How much exercise do I need?"
- "What are the warning signs of a heart attack?"
- "What are the main risk factors?"

**To enable advanced AI features:**
1. Install: `pip install google-genai`
2. Get free API key from https://makersuite.google.com/app/apikey
3. Add to `.streamlit/secrets.toml`: `GEMINI_API_KEY = "your-key"`

⚠️ *Medical Disclaimer: General health information only. Not a substitute for professional medical advice. Always consult your doctor.*"""
    
    def get_status(self):
        """Return the current status of the AI assistant"""
        return {
            "available": self.is_available,
            "message": self.error_message if not self.is_available else "Gemini AI connected and ready! (FREE)"
        }