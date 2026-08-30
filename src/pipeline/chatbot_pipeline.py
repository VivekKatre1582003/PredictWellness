import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
from src.logger import logging

# Find .env path relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

MEDICAL_SYSTEM_INSTRUCTION = """
You are "PredictWellness AI", an advanced, empathetic, and knowledgeable AI Medical and Health Companion for the PredictWellness platform.

Your primary mission:
1. Provide accurate, evidence-based health, medical, and wellness guidance in clear, easy-to-understand language.
2. Explain medical conditions, symptoms, and risk factors related to:
   - Heart Disease (cholesterol, resting blood pressure, angina, ECG, thalassemia)
   - Diabetes (glucose levels, insulin resistance, HbA1c, BMI, diet)
   - Liver Disease (bilirubin, SGOT/SGPT, albumin, ALP, liver enzymes)
   - Stroke (hypertension, ischemic/hemorrhagic stroke signs, prevention)
   - General wellness, nutrition, fitness, sleep, and preventative health.
3. If the user shares prediction results or health metrics from PredictWellness tests, interpret the numbers calmly, explain what they mean, and suggest actionable lifestyle modifications.

Communication Style:
- Professional, supportive, and compassionate.
- Use clear formatting with bullet points, bold key terms, and short paragraphs for readability.
- Avoid overly dense clinical jargon without providing a simple explanation.

Crucial Safety & Ethical Guardrails:
- ALWAYS emphasize that your answers are for educational and informational purposes only and do NOT replace professional clinical diagnosis, treatment, or advice from a licensed physician.
- If a user mentions emergency warning signs (such as acute chest pain radiating to the arm/jaw, sudden slurred speech or facial drooping, difficulty breathing, sudden severe headache, or loss of consciousness), IMMEDIATELY advise them to call emergency services (e.g., 911, 112, 999) or visit the nearest emergency room.
"""

class ChatbotPipeline:
    def __init__(self):
        # Preferred models starting with gemini-3.6-flash
        self.preferred_models = [
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash-lite",
            "gemini-pro-latest"
        ]

    def _get_api_key(self) -> str:
        """Fetch API key from system environment or project .env."""
        # 1. Check system environment first (primary source for cloud platforms like Render)
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key and env_key != "your_gemini_api_key_here" and "your_key" not in env_key:
            return env_key

        # 2. Check local .env file as fallback (for local development)
        if ENV_PATH.exists():
            try:
                vals = dotenv_values(ENV_PATH)
                key = vals.get("GEMINI_API_KEY", "").strip()
                if key and key != "your_gemini_api_key_here" and "your_key" not in key:
                    return key
            except Exception as e:
                logging.warning(f"Failed to read local .env values: {e}")
        
        # 3. Reload dotenv as last resort
        load_dotenv(override=True)
        env_key_fallback = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key_fallback and env_key_fallback != "your_gemini_api_key_here" and "your_key" not in env_key_fallback:
            return env_key_fallback

        return None

    def get_response(self, user_message: str, history: list = None, context: dict = None) -> str:
        """
        Generates a medical AI response for user_message considering history and optional context.
        """
        api_key = self._get_api_key()

        if not api_key:
            return (
                "⚠️ **Gemini API Key Required**\n\n"
                "To activate PredictWellness AI, please provide your Google Gemini API key in the `.env` file:\n\n"
                "1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).\n"
                "2. Open the `.env` file in the project folder.\n"
                "3. Set `GEMINI_API_KEY=your_actual_api_key` and save.\n\n"
                "Once saved, ask your question again to start chatting!"
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # Build context prompt if provided (e.g. from prediction page)
            augmented_message = user_message
            if context and isinstance(context, dict):
                disease = context.get('disease')
                prediction = context.get('prediction')
                if disease and prediction is not None:
                    status = "Higher Risk" if str(prediction) == "1" else "Low Risk"
                    augmented_message = (
                        f"[Context: User just received a {disease.capitalize()} prediction result of '{status}']\n\n"
                        f"User Query: {user_message}"
                    )

            # Build contents from conversation history
            contents = []
            if history and isinstance(history, list):
                for turn in history:
                    role = turn.get('role', 'user')
                    # map 'assistant' or 'bot' to 'model' for Gemini SDK
                    if role in ['assistant', 'bot', 'model']:
                        sdk_role = 'model'
                    else:
                        sdk_role = 'user'
                    
                    text_content = turn.get('content') or turn.get('text') or ''
                    if text_content.strip():
                        contents.append(
                            types.Content(
                                role=sdk_role,
                                parts=[types.Part.from_text(text=text_content)]
                            )
                        )

            # Append current user query
            contents.append(
                types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=augmented_message)]
                )
            )

            # Configure Generation Config
            config = types.GenerateContentConfig(
                system_instruction=MEDICAL_SYSTEM_INSTRUCTION,
                temperature=0.3,  # Controlled & accurate for medical guidance
                top_p=0.9,
                max_output_tokens=1024,
            )

            last_exception = None

            for model in self.preferred_models:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config
                    )
                    if response and response.text:
                        return response.text
                except Exception as model_err:
                    last_exception = model_err
                    logging.warning(f"Attempt with model {model} failed: {str(model_err)}. Trying next candidate...")
                    continue

            if last_exception:
                raise last_exception

            return "I apologize, but I couldn't generate a response right now. Please try asking again."

        except Exception as e:
            err_msg = str(e)
            logging.error(f"Error generating chatbot response: {err_msg}")
            if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                return (
                    "⚠️ **Invalid Gemini API Key**\n\n"
                    "The provided `GEMINI_API_KEY` was not recognized. "
                    "Please check your API key in the `.env` file."
                )
            elif "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                return (
                    "⏳ **Quota Limit Reached**\n\n"
                    "Your Gemini API rate limit has been momentarily reached. Please wait a minute and try again."
                )
            return f"❌ **An error occurred while consulting PredictWellness AI**: {err_msg}"
