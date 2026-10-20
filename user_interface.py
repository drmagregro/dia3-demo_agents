import streamlit as st
from llm_agent import ConversationAgent


class UserInterface:
	def __init__(self):
		st.set_page_config(page_title="Jarvis")
		st.title("Barbe-Quasar")
		self.conversation_agent_object = self.get_persistant_conversation_agent()


	def get_persistant_conversation_agent(self):
		if "conversation_agent_object" not in st.session_state:
			st.session_state.conversation_agent_object = ConversationAgent()
		return st.session_state.conversation_agent_object


	def show_discussion_history(self):
		for message in self.conversation_agent_object.get_conversation_history():
			with st.chat_message(message["role"]):
				st.markdown(message["content"])


	def build_question(self, user_input):
		agent = self.conversation_agent_object

		# 1. Question vocale ou écrite
		if user_input.audio:
			question = agent.transcribe_audio(user_input.audio)
		else:
			question = user_input.text or ""

		# 2. Fichiers joints : images ou texte
		for file in user_input.files:
			if file.type.startswith("image/"):
				description = agent.describe_image(file, question)
				question += f"\n\n--- Image : {file.name} ---\n{description}"
			else:
				content = agent.read_uploaded_file(file)
				question += f"\n\n--- Fichier : {file.name} ---\n{content}"

		return question


	def run(self):
		user_input = st.chat_input(
			"Que veux tu moussaillon ?",
			accept_file=True,
			file_type=["txt", "md", "csv", "pdf", "png", "jpg", "jpeg", "webp"],
			accept_audio=True
		)

		if user_input:
			question = self.build_question(user_input)
			if question.strip():
				with st.spinner("Barbe-Quasar réfléchit..."):
					self.conversation_agent_object.ask_llm(question)

		self.show_discussion_history()


if __name__ == "__main__":
	user_interface_object = UserInterface()
	user_interface_object.run()