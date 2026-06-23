# ============================================================
# app.py — Creator AI Dashboard VERSION 4
# PAGES:
# 1. Home           — scoreboard + API key input
# 2. My Channel     — channel analysis + outlier detector
# 3. Competitor     — 10 competitor @handles + trend stealing
# 4. Virality       — predictor + AI title scorer (Groq)
# 5. Trend Analysis — category trends from dataset
# 6. Retention      — placeholder
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
import requests
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from datetime import datetime
import os
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="Creator AI — YouTube Intelligence",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# LOAD MODEL + DATA FROM MYSQL
# ============================================================

@st.cache_resource
def load_model():
    # Load the trained Random Forest model saved from Jupyter notebook
    with open("models/virality_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_columns():
    # Load exact feature list model was trained on — order matters
    with open("models/feature_columns.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_data():
    # Connect to Aiven cloud MySQL and load video dataset
    conn = mysql.connector.connect(
        host=os.getenv("AIVEN_HOST"),
        port=int(os.getenv("AIVEN_PORT", 25482)),
        user=os.getenv("AIVEN_USER"),
        password=os.getenv("AIVEN_PASSWORD"),
        database=os.getenv("AIVEN_DB"),
        ssl_ca="ca.pem",
        ssl_verify_cert=True
    )
    df = pd.read_sql("SELECT * FROM videos", conn)
    conn.close()
    df['published_at']    = pd.to_datetime(df['published_at'])
    df['upload_hour']     = df['published_at'].dt.hour
    df['upload_day']      = df['published_at'].dt.dayofweek
    df['date']            = df['published_at'].dt.date
    df['is_viral']        = (df['views'] >= 1_000_000).astype(int)
    df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
    df['like_rate']       = df['likes']    / (df['views'] + 1)
    df['comment_rate']    = df['comments'] / (df['views'] + 1)
    return df

model           = load_model()
FEATURE_COLUMNS = load_feature_columns()
df_full         = load_data()

# ============================================================
# SCOREBOARD — stored in Aiven MySQL
# ============================================================

def get_db_conn():
    return mysql.connector.connect(
        host=os.getenv("AIVEN_HOST"),
        port=int(os.getenv("AIVEN_PORT", 25482)),
        user=os.getenv("AIVEN_USER"),
        password=os.getenv("AIVEN_PASSWORD"),
        database=os.getenv("AIVEN_DB"),
        ssl_ca="ca.pem",
        ssl_verify_cert=True
    )

def get_scoreboard():
    try:
        conn   = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scoreboard (
                metric VARCHAR(50) PRIMARY KEY,
                value  INT DEFAULT 0
            )
        """)
        for metric in ['channels_analyzed', 'videos_scanned', 'competitors_tracked']:
            cursor.execute(
                "INSERT IGNORE INTO scoreboard (metric, value) VALUES (%s, 0)", (metric,)
            )
        conn.commit()
        cursor.execute("SELECT metric, value FROM scoreboard")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except:
        return {'channels_analyzed': 0, 'videos_scanned': 0, 'competitors_tracked': 0}

def update_scoreboard(channels=0, videos=0, competitors=0):
    try:
        conn   = get_db_conn()
        cursor = conn.cursor()
        if channels:
            cursor.execute(
                "UPDATE scoreboard SET value = value + %s WHERE metric = 'channels_analyzed'",
                (channels,)
            )
        if videos:
            cursor.execute(
                "UPDATE scoreboard SET value = value + %s WHERE metric = 'videos_scanned'",
                (videos,)
            )
        if competitors:
            cursor.execute(
                "UPDATE scoreboard SET value = value + %s WHERE metric = 'competitors_tracked'",
                (competitors,)
            )
        conn.commit()
        conn.close()
    except:
        pass

# ============================================================
# SAVE USER SCAN DATA TO AIVEN — for future model retraining
# Why: Every competitor scan fetches real YouTube videos.
# We save them to grow our dataset automatically over time.
# User's API key is NEVER saved — only the video data.
# ============================================================

def save_videos_to_db(df_videos):
    try:
        conn   = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id        VARCHAR(20) PRIMARY KEY,
                title           TEXT,
                channel         VARCHAR(255),
                published_at    DATETIME,
                category_id     INT,
                category_name   VARCHAR(100),
                region          VARCHAR(10),
                tags            INT,
                description_len INT,
                views           BIGINT,
                likes           BIGINT,
                comments        BIGINT,
                upload_hour     INT,
                upload_day      INT,
                engagement_rate FLOAT,
                like_rate       FLOAT,
                comment_rate    FLOAT,
                title_length    INT,
                title_word_count INT,
                has_number      INT,
                has_caps_word   INT,
                has_emoji       INT,
                has_tags        INT,
                tag_count_bucket FLOAT,
                is_viral        INT
            )
        """)
        saved = 0
        for _, row in df_videos.iterrows():
            try:
                cursor.execute("""
                    INSERT IGNORE INTO videos
                    (video_id, title, channel, published_at, tags, description_len,
                     views, likes, comments, upload_hour, upload_day,
                     engagement_rate, like_rate, comment_rate,
                     title_length, title_word_count, has_number,
                     has_caps_word, has_emoji, has_tags, tag_count_bucket, is_viral)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    row.get('video_id',''),
                    str(row.get('title','')),
                    str(row.get('channel','')),
                    row.get('published_at', datetime.now()),
                    int(row.get('tag_count', 0)),
                    int(row.get('description_len', 0)),
                    int(row.get('views', 0)),
                    int(row.get('likes', 0)),
                    int(row.get('comments', 0)),
                    int(row.get('upload_hour', 0)),
                    int(row.get('upload_day', 0)),
                    float(row.get('engagement_rate', 0)),
                    float(row.get('like_rate', 0)),
                    float(row.get('comment_rate', 0)),
                    int(row.get('title_length', 0)),
                    int(row.get('title_word_count', 0)),
                    int(row.get('has_number', 0)),
                    int(row.get('has_caps_word', 0)),
                    int(row.get('has_emoji', 0)),
                    int(row.get('has_tags', 0)),
                    float(row.get('tag_count_bucket', 0)),
                    int(row.get('is_viral', 0))
                ))
                saved += 1
            except:
                continue
        conn.commit()
        conn.close()
        return saved
    except:
        return 0

# ============================================================
# YOUTUBE API HELPERS
# ============================================================

def get_youtube_client(api_key):
    return build('youtube', 'v3', developerKey=api_key)

def handle_to_channel_id(youtube, handle):
    try:
        handle   = handle.strip().lstrip('@')
        response = youtube.search().list(
            part="snippet", q=handle, type="channel", maxResults=1
        ).execute()
        if response['items']:
            return response['items'][0]['snippet']['channelId']
        return None
    except:
        return None

def fetch_channel_videos(youtube, channel_id, max_videos=30):
    try:
        channel_resp = youtube.channels().list(
            part="contentDetails,snippet,statistics", id=channel_id
        ).execute()
        if not channel_resp['items']:
            return None, None

        channel_info     = channel_resp['items'][0]
        uploads_playlist = channel_info['contentDetails']['relatedPlaylists']['uploads']
        channel_name     = channel_info['snippet']['title']
        channel_subs     = int(channel_info['statistics'].get('subscriberCount', 0))

        video_ids = []
        next_page = None
        while len(video_ids) < max_videos:
            playlist_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page
            ).execute()
            for item in playlist_resp['items']:
                video_ids.append(item['contentDetails']['videoId'])
            next_page = playlist_resp.get('nextPageToken')
            if not next_page:
                break

        videos = []
        for i in range(0, len(video_ids), 50):
            batch    = video_ids[i:i+50]
            vid_resp = youtube.videos().list(
                part="snippet,statistics", id=','.join(batch)
            ).execute()
            for item in vid_resp['items']:
                snippet   = item['snippet']
                stats     = item.get('statistics', {})
                tag_count = len(snippet.get('tags', []))
                title     = snippet.get('title', '')
                desc      = snippet.get('description', '')
                views     = int(stats.get('viewCount',    0))
                likes     = int(stats.get('likeCount',    0))
                comments  = int(stats.get('commentCount', 0))
                pub_at    = pd.to_datetime(snippet.get('publishedAt', ''))

                videos.append({
                    'video_id'        : item['id'],
                    'title'           : title,
                    'published_at'    : pub_at,
                    'views'           : views,
                    'likes'           : likes,
                    'comments'        : comments,
                    'tags'            : snippet.get('tags', []),
                    'tag_count'       : tag_count,
                    'description'     : desc,
                    'description_len' : len(desc),
                    'channel'         : channel_name,
                    'url'             : f"https://youtube.com/watch?v={item['id']}",
                    'upload_hour'     : pub_at.hour,
                    'upload_day'      : pub_at.dayofweek,
                    'title_length'    : len(title),
                    'title_word_count': len(title.split()),
                    'has_number'      : int(bool(re.search(r'\d', title))),
                    'has_caps_word'   : int(bool(re.search(r'\b[A-Z]{2,}\b', title))),
                    'has_emoji'       : int(any(ord(c) > 127 for c in title)),
                    'has_tags'        : int(tag_count > 0),
                    'tag_count_bucket': float(pd.cut([tag_count], bins=[0,5,10,20,50], labels=[1,2,3,4]).astype(float)[0]) if tag_count > 0 else 0.0,
                    'engagement_rate' : (likes + comments) / (views + 1),
                    'like_rate'       : likes    / (views + 1),
                    'comment_rate'    : comments / (views + 1),
                    'outlier_score'   : 0,
                    'is_viral'        : int(views >= 1_000_000)
                })

        df = pd.DataFrame(videos)
        if df.empty:
            return None, None

        mean_views              = df['views'].mean()
        df['outlier_score']     = df['views'] / (mean_views + 1)

        channel_meta = {
            'name'                : channel_name,
            'subs'                : channel_subs,
            'avg_views'           : int(df['views'].mean()),
            'total_videos_fetched': len(df)
        }
        return df, channel_meta

    except Exception as e:
        st.error(f"API Error: {e}")
        return None, None

# ============================================================
# GROQ AI TITLE SCORER
# Why Groq: Free, fast, runs llama3 in cloud — no local setup
# Replaces Ollama which only worked on localhost
# ============================================================

def score_title_with_groq(title, category):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""You are a YouTube growth expert who specializes in viral titles.
A creator is uploading a {category} video with this title: "{title}"

1. Score this title out of 10 for viral potential (be honest)
2. Explain in 2 sentences exactly WHY it scored that way
3. Give 3 rewritten versions that would score higher

Format EXACTLY like this:
SCORE: X/10
REASON: [2 sentence explanation]

REWRITE 1: [rewritten title]
WHY: [one line]

REWRITE 2: [rewritten title]
WHY: [one line]

REWRITE 3: [rewritten title]
WHY: [one line]"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)} — Check GROQ_API_KEY in .env"

# ============================================================
# CONSTANTS
# ============================================================

CATEGORY_MAP = {
    "Music"         : 10,
    "Gaming"        : 20,
    "Entertainment" : 24,
    "Howto & Style" : 26,
    "Science & Tech": 28,
    "People & Blogs": 22,
    "Sports"        : 17,
    "News"          : 25
}
CATEGORY_NAME_CODES = {
    "Entertainment" : 0,
    "Gaming"        : 1,
    "Howto & Style" : 2,
    "Music"         : 3,
    "News"          : 4,
    "People & Blogs": 5,
    "Science & Tech": 6,
    "Sports"        : 7
}
REGION_MAP = {"India (IN)": 0, "USA (US)": 1, "UK (GB)": 2}
DAY_NAMES  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Creator AI")
st.sidebar.markdown("YouTube Intelligence Engine")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 YouTube API Key")
api_key = st.sidebar.text_input(
    "Paste your API key here",
    type="password",
    placeholder="AIza...",
    help="Get free key from console.cloud.google.com"
)
api_key_valid = False
if api_key:
    try:
        test_youtube = build('youtube', 'v3', developerKey=api_key)
        test_youtube.videos().list(part="snippet", id="dQw4w9WgXcQ").execute()
        st.sidebar.success("✅ API Key verified and ready")
        api_key_valid = True
    except:
        st.sidebar.error("❌ Invalid API key — check and try again")
else:
    st.sidebar.warning("⚠️ Add API key to use Channel & Competitor pages")

st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📺 My Channel Analyzer",
    "🕵️ Competitor Tracker",
    "🔮 Virality Predictor",
    "📊 Trend Analysis",
    "📉 Retention Analyzer"
])

# ============================================================
# PAGE 1 — HOME
# ============================================================

if page == "🏠 Home":
    st.title("🎯 Creator AI — YouTube Intelligence Engine")
    st.markdown("### From zero subscribers to viral — powered by data")
    st.markdown("---")

    scores = get_scoreboard()

    st.markdown("### 🏆 Live Usage Scoreboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📹 Dataset Videos",      f"{len(df_full):,}")
    c2.metric("🔥 Viral in Dataset",    f"{df_full['is_viral'].sum():,}")
    c3.metric("📺 Channels Analyzed",   f"{scores.get('channels_analyzed',  0):,}")
    c4.metric("🎬 Videos Scanned",      f"{scores.get('videos_scanned',     0):,}")
    c5.metric("🕵️ Competitors Tracked", f"{scores.get('competitors_tracked',0):,}")

    st.markdown(f"*Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}*")
    st.markdown("---")
    st.success(f"✅ Live data from MySQL — {len(df_full):,} videos across {df_full['category_name'].nunique()} categories and {df_full['region'].nunique()} regions")

    st.markdown("### What Creator AI does for you")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📺 **My Channel Analyzer**\nEnter your @handle → see your last 30 videos, which ones overperformed, and what topics to make next")
    with c2:
        st.info("🕵️ **Competitor Tracker**\nTrack up to 10 competitors → find their outlier videos → steal their best topics before they repeat")
    with c3:
        st.info("🔮 **Virality Predictor**\nType your title and description → get viral probability + AI rewrites your title to rank higher")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.info("📊 **Trend Analysis**\nSee which categories are growing, what upload times work, and where the views are")
    with c5:
        st.info("📉 **Retention Analyzer**\nUpload your YouTube Studio CSV → see exactly where viewers drop off *(coming soon)*")
    with c6:
        st.info("🔑 **Bring Your Own API Key**\nPaste your free YouTube API key in the sidebar — no monthly subscription, no ₹3000 fees")

# ============================================================
# PAGE 2 — MY CHANNEL ANALYZER
# ============================================================

elif page == "📺 My Channel Analyzer":
    st.title("📺 My Channel Analyzer")
    st.markdown("Enter your YouTube @handle to analyze your last 30 videos")

    if not api_key_valid:
        st.error("⚠️ Please add your YouTube API key in the sidebar first")
        st.stop()

    handle = st.text_input("Your YouTube Channel Handle", placeholder="@YourChannelName")

    if st.button("🔍 Analyze My Channel", use_container_width=True, disabled=not handle):
        with st.spinner("Fetching your last 30 videos from YouTube..."):
            youtube    = get_youtube_client(api_key)
            channel_id = handle_to_channel_id(youtube, handle)
            if not channel_id:
                st.error(f"Could not find channel: {handle}")
                st.stop()
            df_ch, meta = fetch_channel_videos(youtube, channel_id, max_videos=30)
            if df_ch is None:
                st.error("Could not fetch videos. Check your API key and handle.")
                st.stop()

        update_scoreboard(channels=1, videos=len(df_ch))

        st.markdown("---")
        st.markdown(f"### 📺 {meta['name']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Subscribers",     f"{meta['subs']:,}")
        col2.metric("Videos Fetched",  f"{meta['total_videos_fetched']}")
        col3.metric("Avg Views",       f"{meta['avg_views']:,}")
        col4.metric("Viral Threshold", "1,000,000 views")

        st.markdown("---")
        st.markdown("### ⏰ Your Best Upload Time")
        hour_perf = df_ch.groupby('upload_hour')['views'].mean().reset_index()
        day_perf  = df_ch.groupby('upload_day')['views'].mean().reset_index()
        day_perf['Day'] = day_perf['upload_day'].map(lambda x: DAY_NAMES[x])

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(hour_perf, x='upload_hour', y='views',
                         title="Your Avg Views by Upload Hour",
                         color='views', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(day_perf, x='Day', y='views',
                          title="Your Avg Views by Upload Day",
                          color='views', color_continuous_scale='Greens')
            st.plotly_chart(fig2, use_container_width=True)

        best_hour = hour_perf.loc[hour_perf['views'].idxmax(), 'upload_hour']
        best_day  = day_perf.loc[day_perf['views'].idxmax(), 'Day']
        st.success(f"✅ Your best time to upload: **{best_day}** at **{int(best_hour)}:00**")

        st.markdown("---")
        st.markdown("### 🔥 Your Outlier Videos (Overperformed Your Average)")
        outliers = df_ch[df_ch['outlier_score'] >= 1.5].sort_values('outlier_score', ascending=False)

        if outliers.empty:
            st.info("No strong outliers yet — keep uploading consistently")
        else:
            for _, row in outliers.iterrows():
                with st.expander(f"🔥 {row['title']} — {row['views']:,} views ({row['outlier_score']:.1f}x your average)"):
                    st.markdown(f"**Views:** {row['views']:,} | **Likes:** {row['likes']:,}")
                    st.markdown(f"**Uploaded:** {row['published_at'].strftime('%d %b %Y')}")
                    tags_str = ', '.join(row['tags'][:10]) if row['tags'] else 'None'
                    st.markdown(f"**Tags used:** {tags_str}")
                    st.markdown(f"**[Watch Video]({row['url']})**")

        st.markdown("---")
        st.markdown("### 💡 Topics You Should Make More Of")
        top_videos = df_ch.nlargest(5, 'views')[['title', 'views', 'url']]
        for _, row in top_videos.iterrows():
            st.markdown(f"- **[{row['title']}]({row['url']})** — {row['views']:,} views")

        st.markdown("---")
        st.markdown("### 📋 All Your Videos")
        st.dataframe(
            df_ch[['title','views','likes','comments','upload_hour','outlier_score']]\
                .sort_values('views', ascending=False),
            use_container_width=True
        )

# ============================================================
# PAGE 3 — COMPETITOR TRACKER
# ============================================================

elif page == "🕵️ Competitor Tracker":
    st.title("🕵️ Competitor Tracker")
    st.markdown("Track up to 10 competitors → find their outlier videos → steal their best topics")

    if not api_key_valid:
        st.error("⚠️ Please add your YouTube API key in the sidebar first")
        st.stop()

    # Consent checkbox — user agrees before data is saved
    save_consent = st.checkbox(
        "✅ Help improve Creator AI (anonymously save scanned video data to improve the model)",
        value=False
    )

    st.markdown("### Enter competitor @handles (one per box)")
    handles = []
    cols = st.columns(2)
    for i in range(10):
        with cols[i % 2]:
            h = st.text_input(f"Competitor {i+1}", placeholder=f"@CompetitorHandle", key=f"comp_{i}")
            if h.strip():
                handles.append(h.strip())

    if st.button("🔍 Scan All Competitors", use_container_width=True, disabled=len(handles) == 0):
        all_videos = []
        progress   = st.progress(0)
        status     = st.empty()

        for idx, handle in enumerate(handles):
            status.text(f"Scanning {handle}... ({idx+1}/{len(handles)})")
            youtube    = get_youtube_client(api_key)
            channel_id = handle_to_channel_id(youtube, handle)
            if not channel_id:
                st.warning(f"Could not find: {handle} — skipping")
                continue
            df_ch, meta = fetch_channel_videos(youtube, channel_id, max_videos=30)
            if df_ch is not None:
                df_ch['handle'] = handle
                all_videos.append(df_ch)
                update_scoreboard(competitors=1, videos=len(df_ch))

                # Save to DB only if user consented
                if save_consent:
                    saved = save_videos_to_db(df_ch)
                    st.toast(f"💾 Saved {saved} new videos from {handle} to dataset")

            progress.progress((idx + 1) / len(handles))

        status.empty()
        progress.empty()

        if not all_videos:
            st.error("Could not fetch any competitor data")
            st.stop()

        df_all = pd.concat(all_videos, ignore_index=True)
        st.markdown("---")
        st.success(f"✅ Scanned {len(handles)} competitors — {len(df_all)} total videos analyzed")

        st.markdown("### ⏰ When Do Your Competitors Post?")
        st.markdown("*Most competitors posting at the same time = your audience is active then. Post 15 minutes earlier.*")
        hour_dist = df_all.groupby('upload_hour').size().reset_index(name='video_count')
        fig = px.bar(hour_dist, x='upload_hour', y='video_count',
                     title="Competitor Videos Posted Per Hour",
                     color='video_count', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        peak_hour = hour_dist.loc[hour_dist['video_count'].idxmax(), 'upload_hour']
        post_at   = (peak_hour - 1) % 24
        st.warning(f"⚡ Competitors mostly post at **{int(peak_hour)}:00** — you post at **{int(post_at)}:45** to get there first")

        st.markdown("---")
        st.markdown("### 🔥 Their Outlier Videos — Topics That Worked For Them")
        outliers_all = df_all[df_all['outlier_score'] >= 2.0].sort_values('outlier_score', ascending=False)
        if outliers_all.empty:
            st.info("No strong outliers found across competitors")
        else:
            for _, row in outliers_all.head(15).iterrows():
                with st.expander(f"🔥 [{row['handle']}] {row['title']} — {row['views']:,} views ({row['outlier_score']:.1f}x their average)"):
                    st.markdown(f"**Channel:** {row['channel']}")
                    st.markdown(f"**Views:** {row['views']:,} | **Likes:** {row['likes']:,}")
                    st.markdown(f"**Posted:** {row['published_at'].strftime('%d %b %Y')} at {row['upload_hour']}:00")
                    tags_preview = ', '.join(row['tags'][:10]) if isinstance(row['tags'], list) else 'None visible'
                    st.markdown(f"**Tags they used:** {tags_preview}")
                    st.markdown(f"**[Watch Video]({row['url']})**")

        st.markdown("---")
        st.markdown("### 🔑 Keywords Trending in Competitor Titles")
        all_words  = ' '.join(df_all['title'].tolist()).lower().split()
        stop_words = {'the','a','an','in','on','at','to','for','of','and','or','is',
                      'i','my','your','this','that','with','how','why','when','what',
                      'it','its','was','are','be','been','have','has','do','did',
                      'will','can','just','we','they','he','she','me','new','get'}
        word_freq           = pd.Series(all_words).value_counts().reset_index()
        word_freq.columns   = ['keyword', 'count']
        word_freq           = word_freq[~word_freq['keyword'].isin(stop_words)].head(20)
        fig2 = px.bar(word_freq, x='count', y='keyword', orientation='h',
                      title="Top 20 Keywords in Competitor Titles",
                      color='count', color_continuous_scale='Oranges')
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 All Competitor Videos")
        st.dataframe(
            df_all[['handle','title','views','likes','comments','upload_hour','outlier_score']]\
                .sort_values('views', ascending=False),
            use_container_width=True
        )

# ============================================================
# PAGE 4 — VIRALITY PREDICTOR + AI TITLE SCORER
# ============================================================

elif page == "🔮 Virality Predictor":
    st.title("🔮 Virality Predictor + AI Title Scorer")
    st.markdown("Type your actual title and description — we calculate everything automatically")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Your Video Details")
        user_title  = st.text_input("Your Video Title", placeholder="e.g. I tried every budget phone under ₹10,000 🔥")
        user_desc   = st.text_area("Your Video Description", placeholder="Paste your description here...", height=150)
        category    = st.selectbox("Video Category", list(CATEGORY_MAP.keys()))
        region      = st.selectbox("Target Region",  list(REGION_MAP.keys()))
        upload_day  = st.selectbox("Upload Day", DAY_NAMES)
        upload_hour = st.slider("What time will you post? (e.g. 18 = 6pm)", 0, 23, 18)
        tag_count   = st.slider("How many hashtags/tags will you add?", 0, 50, 15)

    with col2:
        st.markdown("### 📊 Auto-Calculated from Your Title")
        if user_title:
            t_len   = len(user_title)
            t_words = len(user_title.split())
            t_num   = int(bool(re.search(r'\d', user_title)))
            t_caps  = int(bool(re.search(r'\b[A-Z]{2,}\b', user_title)))
            t_emoji = int(any(ord(c) > 127 for c in user_title))
            d_len   = len(user_desc)

            st.metric("Title Length",       f"{t_len} characters")
            st.metric("Word Count",         f"{t_words} words")
            st.metric("Has Number?",        "✅ Yes" if t_num   else "❌ No")
            st.metric("Has CAPS Word?",     "✅ Yes" if t_caps  else "❌ No")
            st.metric("Has Emoji?",         "✅ Yes" if t_emoji else "❌ No")
            st.metric("Description Length", f"{d_len} characters")

            st.markdown("### 💡 Quick Tips")
            if t_len < 40:
                st.warning("📌 Title too short — aim for 50–70 characters")
            elif t_len > 80:
                st.warning("📌 Title too long — trim to under 70 characters")
            else:
                st.success("✅ Title length looks good")
            if not t_emoji:
                st.info("📌 Adding an emoji can increase click-through rate")
            if tag_count < 10:
                st.info("📌 Viral videos average 15+ tags")
        else:
            st.info("Type your title on the left to see auto-analysis here")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_clicked = st.button("🚀 Predict Virality",    use_container_width=True, disabled=not user_title)
    with col_btn2:
        score_clicked   = st.button("🤖 Score Title with AI", use_container_width=True, disabled=not user_title)

    if predict_clicked and user_title:
        t_len   = len(user_title)
        t_words = len(user_title.split())
        t_num   = int(bool(re.search(r'\d', user_title)))
        t_caps  = int(bool(re.search(r'\b[A-Z]{2,}\b', user_title)))
        t_emoji = int(any(ord(c) > 127 for c in user_title))
        d_len   = len(user_desc)
        day_map = {d: i for i, d in enumerate(DAY_NAMES)}

        # Build input exactly matching FEATURE_COLUMNS order from training
        input_data = pd.DataFrame([{
            'category_id'          : CATEGORY_MAP[category],
            'category_name_encoded': CATEGORY_NAME_CODES[category],
            'region_encoded'       : REGION_MAP[region],
            'tags'                 : tag_count,
            'description_len'      : d_len,
            'upload_hour'          : upload_hour,
            'upload_day'           : day_map[upload_day],
            'title_length'         : t_len,
            'title_word_count'     : t_words,
            'has_number'           : t_num,
            'has_caps_word'        : t_caps,
            'has_emoji'            : t_emoji,
            'has_tags'             : int(tag_count > 0),
            'tag_count_bucket'     : float(pd.cut([tag_count], bins=[0,5,10,20,50], labels=[1,2,3,4]).astype(float)[0]) if tag_count > 0 else 0.0
        }])

        # Ensure column order matches training exactly
        input_data = input_data[FEATURE_COLUMNS]

        prob     = model.predict_proba(input_data)[0][1]
        is_viral = prob >= 0.5

        if is_viral:
            st.success(f"🔥 HIGH VIRAL POTENTIAL — {prob*100:.1f}% probability")
        else:
            st.warning(f"📉 LOW VIRAL POTENTIAL — {prob*100:.1f}% probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            title={"text": "Viral Probability %"},
            gauge={
                "axis" : {"range": [0, 100]},
                "bar"  : {"color": "green" if is_viral else "red"},
                "steps": [
                    {"range": [0,  50], "color": "#ffcccc"},
                    {"range": [50, 75], "color": "#fff3cc"},
                    {"range": [75,100], "color": "#ccffcc"},
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    if score_clicked and user_title:
        with st.spinner("🧠 Groq AI is analyzing your title... (5–10 seconds)"):
            result = score_title_with_groq(user_title, category)

        st.markdown("### 🤖 AI Title Analysis")
        score_value = None
        for line in result.split('\n'):
            if line.startswith("SCORE:"):
                try:
                    score_value = float(line.replace("SCORE:", "").strip().split("/")[0])
                except:
                    pass
                break

        if score_value is not None:
            col_s, col_p = st.columns([1, 3])
            with col_s:
                st.metric("AI Viral Score", f"{score_value}/10")
            with col_p:
                st.progress(score_value / 10)

        st.markdown(result)

# ============================================================
# PAGE 5 — TREND ANALYSIS
# ============================================================

elif page == "📊 Trend Analysis":
    st.title("📊 Trend Analysis")
    st.markdown("Category and engagement trends from dataset")

    cat_views = df_full.groupby('category_name')['views'].mean().reset_index().sort_values('views', ascending=False)
    fig1 = px.bar(cat_views, x='category_name', y='views',
                  title="Average Views by Category",
                  color='views', color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df_full, x='views', nbins=50,
                        title="Distribution of Views Across All Videos",
                        color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig2, use_container_width=True)

    viral_cat = df_full.groupby('category_name')['is_viral'].mean().reset_index()
    viral_cat.columns = ['Category', 'Viral Rate']
    fig3 = px.bar(viral_cat.sort_values('Viral Rate', ascending=False),
                  x='Category', y='Viral Rate',
                  title="Viral Rate by Category",
                  color='Viral Rate', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 🏆 Top 10 Most Viewed Videos in Dataset")
    top10 = df_full.nlargest(10, 'views')[['title','channel','category_name','views','likes']]
    st.dataframe(top10, use_container_width=True)

# ============================================================
# PAGE 6 — RETENTION ANALYZER (PLACEHOLDER)
# ============================================================

elif page == "📉 Retention Analyzer":
    st.title("📉 Retention Analyzer")
    st.markdown("See exactly where your viewers drop off")

    st.info("""
### 🚧 Coming Soon

**How it will work:**
1. Go to **YouTube Studio → Analytics → Audience Retention**
2. Select any video → click **Export** → download CSV
3. Upload that CSV here
4. We show you exactly which second viewers drop off and why

**Why YouTube doesn't give this through API:**
Retention data is private — only you can see it in YouTube Studio.
No third party app (not even VidIQ) can access it without you exporting it manually.
    """)

    st.markdown("---")
    st.file_uploader("Upload your YouTube Studio Retention CSV (coming soon)", type=['csv'], disabled=True)

    st.markdown("### 📋 What you will see here:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📉 **Drop-off curve**\nExact second where viewers leave")
    with c2:
        st.info("🔁 **Re-watch spikes**\nMoments viewers replay — your best content")
    with c3:
        st.info("💡 **Fix suggestions**\nWhat to change in your next video based on retention")