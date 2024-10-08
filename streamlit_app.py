import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

# Initialize Vertex AI
vertexai.init(project="personalgcp-438016", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-002")

# Safety settings and generation config for Gemini
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

def generate_summary(video_url):
    try:
        video_part = Part.from_uri(mime_type="video/*", uri=video_url)
    except Exception as e:
        st.write(f"Error processing video URL: {e}")
        return None

    responses = model.generate_content(
        [video_part, "Please summarize the main points of this video."],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    summary = ""
    for response in responses:
        summary += response.text

    return summary

def main():
    st.title("YouTube Video Conversation")

    # Initialize session state variables
    if "summary" not in st.session_state:
        st.session_state.summary = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Input for the YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL")

    # Button to generate the summary
    if st.button("Generate Summary"):
        summary = generate_summary(video_url)
        if summary:
            st.session_state.summary = summary  # Store the summary in session state
            st.session_state.conversation = []  # Reset conversation when a new video is summarized

    # Display the summary if it exists
    if st.session_state.summary:
        st.write("### Summary:")
        st.write(st.session_state.summary)

        # Display existing conversation
        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for user to ask questions about the video
        user_input = st.chat_input("Ask a question about the video:")
        if user_input:
            # Store user's message
            st.session_state.conversation.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate assistant's response
            responses = model.generate_content(
                [f"Video summary: {st.session_state.summary}", user_input],
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )

            # Collect responses into a single string
            assistant_response = ""
            for response in responses:
                assistant_response += response.text

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # Store assistant's response
            st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()
