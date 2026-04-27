import os
import logging
import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

EXPLAIN_PROMPT = """You are a salary expert. Explain in 2-3 sentences why someone with the following profile would earn the predicted salary.

Profile:
- Job Title: {job_title}
- Location: {location}
- Education: {education}
- Years of Experience: {years_experience}
- Skills: {skills}
- Predicted Salary: ${predicted_salary:,.0f} CAD

Be specific and mention key factors driving the salary. Keep it concise and professional."""


async def get_salary_explanation(
    job_title: str,
    location: str,
    education: str,
    years_experience: int,
    skills: list[str],
    predicted_salary: float,
) -> str:
    prompt = EXPLAIN_PROMPT.format(
        job_title=job_title,
        location=location,
        education=education,
        years_experience=years_experience,
        skills=", ".join(skills) if skills else "None listed",
        predicted_salary=predicted_salary,
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama explanation failed: %s", e)
        return "Explanation unavailable — AI service unreachable."
