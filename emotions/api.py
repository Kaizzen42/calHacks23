from hume import HumeBatchClient
from hume.models.config import LanguageConfig

from keys import HUME_API_KEY
from pathlib import Path
from typing import Any, Dict, List
import Stringifier
import json

# Prompt
TEXT = "I'm so excited for the camping trip next weekend! I hope I don't encounter any bears though. What an amazing experience it might be, to be eaten alive by a bear."
MUSIC_EMOTIONS = ["Anxiety", "Awe", "Contentment", "Craving","Disappointment","Ecstasy","Enthusiasm","Fear",
                  "Horror","Joy", "Love","Nostalgia","Pain","Realization","Sadness","Triumph",
                  "Surprise (negative)", "Surprise (positive)" ]
# Write text to a file
filepath = "emotions/inputs/text.txt"
with open(filepath, "w") as fp:
    fp.write(TEXT)


# Setup HUME client
client = HumeBatchClient(HUME_API_KEY)
config = LanguageConfig(granularity="sentence", identify_speakers=True)

# Submimt a batch job SYnc
job = client.submit_job(None, [config], files=[filepath])
print("Running...", job)
job.await_complete()
print("Job completed with status: ", job.get_status())

# Method to select some emotions
from collections import defaultdict
music_emo_scores = defaultdict(float)
count_dict = defaultdict(int)

# Function to update the sum and count
def update_dict(key, new_value):
    music_emo_scores[key] += new_value
    count_dict[key] += 1

# Function to calculate and return the average
def average(key):
    return music_emo_scores[key] / count_dict[key]


def print_emotions(emotions: List[Dict[str, Any]]) -> None:
    print(f"Raw emotions map: {emotions}")
    emotion_map = {e["name"]: e["score"] for e in emotions}
    # for emotion in ["Excitement", "Joy", "Sadness", "Anger", "Confusion", "Fear"]:
    for emotion in MUSIC_EMOTIONS:
        print(f"- {emotion}: {emotion_map[emotion]:4f}")
        update_dict(emotion, emotion_map[emotion])
# Show scores
emotion_embeddings = []
full_predictions = job.get_predictions()

# Save full predictions
outputFilePath = "emotions/outputs/text.txt"
with open(outputFilePath, "w") as ofp:
    for pred in full_predictions:
        ofp.write(json.dumps(pred))

for source in full_predictions:
    predictions = source["results"]["predictions"]
    for prediction in predictions:
        language_predictions = prediction["models"]["language"]["grouped_predictions"]
        for language_prediction in language_predictions:
            for chunk in language_prediction["predictions"]:
                print(chunk["text"])
                print_emotions(chunk["emotions"])
                emotion_embeddings.append(chunk["emotions"])
                print()        

prompt_add = []
music_emo_scores_avg = {k: average(k) for k in music_emo_scores.keys()}
sorted_music_emo_scores = dict(sorted(music_emo_scores_avg.items(), key=lambda item: item[1], reverse=True))
for k,v in sorted_music_emo_scores.items():
    if v >= 0.5:
        prompt_add.append(f"Extreme {k}")
    elif v >= 0.375:
        prompt_add.append(f"Moderate {k}")    
    elif v >= 0.25:
        prompt_add.append(f"Slight {k}")

print(prompt_add)        
# stringifier = Stringifier.Stringifier()
# for emotion_embedding in emotion_embeddings:
#     emotion_scores = [emotion["score"] for emotion in emotion_embedding]
    
#     text = stringifier.scores_to_text(emotion_scores)
#     print(text)