from hume import HumeBatchClient
from hume.models.config import LanguageConfig

from keys import HUME_API_KEY
from pathlib import Path
from typing import Any, Dict, List
import Stringifier
import json

# Prompt
TEXT = "I'm so excited for the camping trip next weekend! I hope I don't encounter any bears though. What an amazing experience it might be, to be eaten by a bear, alive!"

# Write text to a file
filepath = "demo/inputs/text.txt"
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
def print_emotions(emotions: List[Dict[str, Any]]) -> None:
    emotion_map = {e["name"]: e["score"] for e in emotions}
    for emotion in ["Excitement", "Joy", "Sadness", "Anger", "Confusion", "Fear"]:
        print(f"- {emotion}: {emotion_map[emotion]:4f}")

# Show scores
emotion_embeddings = []
full_predictions = job.get_predictions()

# Save full predictions
outputFilePath = "demo/outputs/text.txt"
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


stringifier = Stringifier.Stringifier()
for emotion_embedding in emotion_embeddings:
    emotion_scores = [emotion["score"] for emotion in emotion_embedding]
    text = stringifier.scores_to_text(emotion_scores)
    print(text)