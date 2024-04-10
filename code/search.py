from googleapiclient.discovery import build
import os
import json
import yaml
import pandas as pd
from datetime import datetime
# Set up YouTube API key
API_KEY = "AIzaSyA2Ghvtqo9_Te5zAjLZIFiJVMzXLCquyIs"

# Function to search for videos related to climate change and extract video IDs
def search_videos(query, max_results=10000):
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Call the search.list method to retrieve search results
    request = youtube.search().list(
        part="id",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    # Extract video IDs from search results
    video_ids = [item['id']['videoId'] for item in response['items']]

    return video_ids

def search(search_query):
    # Perform the search and get video IDs
    video_ids = search_videos(search_query)

    print(f"Number of video IDs related to {search_query}: {len(video_ids)}")

    # Save the video IDs to a JSON file
    output_dir = f"output/search/{search_query.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "videos.json")
    with open(output_file, 'w') as f:
        json.dump(video_ids, f)
    for video_id in video_ids:


        save_video_comments(video_id,output_dir)

    print(f"Video IDs related to {search_query} saved to: {output_file}")


def save_video_comments(video_id,parent_dir):
    # Function to fetch all comments, video ID, video title, and channel name from a YouTube video
    def get_all_video_info(video_id, api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments_info = []

        # Fetch video information
        video_response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        video_title = video_response["items"][0]["snippet"]["title"]
        channel_name = video_response["items"][0]["snippet"]["channelTitle"]
        video_statistics = video_response["items"][0]["statistics"]
        publish_date = video_response["items"][0]["snippet"]["publishedAt"]
        publish_date = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

        return {'VideoID': str(video_id), 'VideoTitle': video_title, 'ChannelName': str(channel_name), 'Statistics': video_statistics, 'PublishDate': publish_date}

    os.makedirs(os.path.join(parent_dir,video_id), exist_ok=True)
    # Fetch video info for the video
    video_info = get_all_video_info(video_id, API_KEY)

    # Save video statistics to a YAML file
    stats_file = os.path.join(parent_dir, video_id, "stats.yaml")
    with open(stats_file, 'w') as yaml_file:
        yaml.dump(video_info, yaml_file, default_flow_style=False)

    print(f"Video statistics saved to: {stats_file}")

    # Function to fetch all comments from a YouTube video
    def get_all_video_comments(video_id, api_key):
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments_info = []

        nextPageToken = None
        while True:
            # Fetch comments in batches of 100
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken
            ).execute()

            for item in response["items"]:
                comment_id = item["snippet"]["topLevelComment"]["id"]
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                comments_info.append({'CommentID': comment_id, 'Comment': comment_text, 'LikeCount': like_count})

                # Check if the comment has replies
                if "replies" in item:
                    for reply_item in item["replies"]["comments"]:
                        reply_id = reply_item["id"]
                        reply_text = reply_item["snippet"]["textDisplay"]
                        reply_like_count = reply_item["snippet"]["likeCount"]
                        comments_info.append({'CommentID': reply_id, 'Comment': reply_text, 'LikeCount': reply_like_count})

            # Check if there are more comments to fetch
            nextPageToken = response.get("nextPageToken")
            if not nextPageToken:
                break

        return comments_info

    # Fetch all comments for the video
    comments_info = get_all_video_comments(video_id, API_KEY)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(parent_dir, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Create a Pandas DataFrame
    df = pd.DataFrame(comments_info)

    # Save the DataFrame as a CSV file in the output directory
    comments_file = os.path.join(output_dir, "comments.csv")
    df.to_csv(comments_file, index=False)

    print(f"Comments saved to: {comments_file}")




if __name__ == '__main__':
    for search_query in ["climate change",'AI','clean energy','GMO']:
        search(search_query)


