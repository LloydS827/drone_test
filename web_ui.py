import streamlit as st
from http import HTTPStatus
import dashscope
import os
import weave

st.title("Drone Video Analysis")

# 获取视频列表
videos_folder = 'videos'
video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# 让用户选择视频
selected_video = st.selectbox("选择要分析的视频", video_files)

# 展示选中的视频
if selected_video:
    video_path = os.path.join(videos_folder, selected_video)
    st.video(video_path)

# 用户输入阿里云Qwen API key
api_key = st.text_input("请输入阿里云Qwen API Key", type="password")
if api_key:
    dashscope.api_key = api_key

# 添加Weave API key配置选项
use_weave = st.checkbox("Using (W&B) Weave for experiment management")
if use_weave:
    weave_api_key = st.text_input("Please enter your Weave API Key", type="password")
    if weave_api_key:
        os.environ["WANDB_API_KEY"] = weave_api_key
        weave.init('video-analysis-example')

# 用户输入问题
input_prompt = st.text_input("Please enter your analysis question", value="Analyze the video content.")

@weave.op()
def analyze_video(video_path):
    """Analyze a single video using the multimodal conversation model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_path},
                {"text": input_prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-max-0809',
        messages=messages
    )
    if response.status_code == HTTPStatus.OK:
        result = response.output.choices[0].message.content[0]['text']
    else:
        result = f"Error: {response.message}"
    return {"prompt": input_prompt, "response": result}

if st.button("Start Analysis"):
    if not api_key:
        st.error("Please enter your Alibaba Cloud Qwen API Key")
    elif use_weave and not weave_api_key:
        st.error("You have selected to use Weave, please enter your Weave API Key")
    elif not selected_video:
        st.error("Please select a video to analyze")
    else:
        with st.spinner("Analyzing video..."):
            video_path = os.path.join(videos_folder, selected_video)
            result = analyze_video(video_path)
        
        st.subheader("Video Analysis Results")
        st.write(f"Question: {result['prompt']}")
        st.write(f"Answer: {result['response']}")

# st.sidebar.info("Note: Please ensure that the 'videos' folder contains the video files to be analyzed.")