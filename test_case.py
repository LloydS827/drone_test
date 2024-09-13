from dotenv import load_dotenv
from http import HTTPStatus
import dashscope
import os
import weave

weave.init('video-analysis-example')

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('DASHSCOPE_API_KEY')

input_prompt = "分析这段视频的拍摄仪器以及拍摄地点"

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
        print(f"Analysis for {video_path}:")
        result = response.output.choices[0].message.content[0]['text']
    else:
        print(f"Error analyzing {video_path}:")
        print(f"Error code: {response.code}")
        print(f"Error message: {response.message}")
        result = f"Error: {response.message}"
    return {"prompt": input_prompt, "response": result}

def analyze_all_videos():
    """Analyze all videos in the 'videos' folder."""
    results = []
    videos_folder = os.path.join(os.getcwd(), 'videos')
    for video_file in os.listdir(videos_folder):
        if video_file.endswith(('.mp4', '.avi', '.mov')):  # Add more video extensions if needed
            video_path = os.path.join(videos_folder, video_file)
            result = analyze_video(video_path)
            results.append(result)
    return results

results = analyze_all_videos()
for result in results:
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}")
    print()