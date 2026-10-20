from groq import Groq
from dotenv import load_dotenv
import base64
import os

CHAT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
AUDIO_MODEL = "whisper-large-v3-turbo"
class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
        self.conversation_history = [
            {
                "role": "system",
                "content": self.read_file(file_path="./context.txt")
            }
        ]


    # ---------- Lecture de fichiers ----------

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


    @staticmethod
    def read_uploaded_file(uploaded_file):
        if uploaded_file.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        return uploaded_file.read().decode("utf-8", errors="ignore")


    # ---------- Audio et images ----------

    def transcribe_audio(self, audio_file):
        transcription = self.groq_client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3-turbo",
            language="fr",
        )
        return transcription.text


    def describe_image(self, image_file, question=""):
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        response = self.groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Décris cette image en détail en français, et retranscris "
                                "tout texte visible. Question de l'utilisateur : " + question
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_file.type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


    # ---------- Mémoire ----------

    def get_conversation_history(self):
        return self.conversation_history[1:]


    def update_conversation_history(self, role, content):
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
            }
        )


    # ---------- LLM ----------

    def ask_llm(self, text_interaction):
        self.update_conversation_history(role="user", content=text_interaction)

        response = self.groq_client.chat.completions.create(
            messages=self.conversation_history,
            model="openai/gpt-oss-120b",
            include_reasoning=False,
        ).choices[0].message.content

        self.update_conversation_history(role="assistant", content=response)
        return response