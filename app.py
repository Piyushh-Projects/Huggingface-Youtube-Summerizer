from urllib.parse import parse_qs, urlparse
import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Sidebar input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# Prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def clean_youtube_url(url):
    if "youtu.be" in url:
        video_id = url.split("/")[-1]
    elif "shorts" in url:
        video_id = url.split("/shorts/")[-1].split("?")[0]
    else:
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get("v")
        video_id = video_id[0] if video_id else None

    if not video_id:
        return None

    return f"https://www.youtube.com/watch?v={video_id}"


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        return None


if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Processing..."):

                docs = []
                clean_url = None

                is_youtube = "youtube.com" in generic_url or "youtu.be" in generic_url

                if is_youtube:
                    clean_url = clean_youtube_url(generic_url)

                    if not clean_url:
                        st.error("Invalid YouTube URL")
                        st.stop()

                    video_id = clean_url.split("v=")[-1]

                    try:
                        loader = YoutubeLoader.from_youtube_url(
                            clean_url,
                            add_video_info=False
                        )
                        docs = loader.load()

                    except Exception:
                        # fallback to transcript
                        text = get_transcript(video_id)

                        if not text:
                            st.error("Could not fetch YouTube transcript")
                            st.stop()

                        docs = [Document(page_content=text)]

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content could be extracted from the URL")
                    st.stop()

                # Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                result = chain.invoke({"input_documents": docs})

                st.success(result["output_text"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")