import numpy as np
from typing import List


class Stringifier:
    RANGES = [(0.26, 0.35), (0.35, 0.44), (0.44, 0.53), (0.53, 0.62), (0.62, 0.71), (0.71, 10)]
    ADVERBS = ["slightly", "somewhat", "moderately", "quite", "very", "extremely"]

    ADJECTIVES_48 = [
        "admiring", "adoring", "appreciative", "amused", "angry", "anxious", "awestruck", "uncomfortable", "bored",
        "calm", "focused", "contemplative", "confused", "contemptuous", "content", "hungry", "determined",
        "disappointed", "disgusted", "distressed", "doubtful", "euphoric", "embarrassed", "disturbed", "entranced",
        "envious", "excited", "fearful", "guilty", "horrified", "interested", "happy", "enamored", "nostalgic",
        "pained", "proud", "inspired", "relieved", "smitten", "sad", "satisfied", "desirous", "ashamed",
        "negatively surprised", "positively surprised", "sympathetic", "tired", "triumphant"
    ]

    ADJECTIVES_53 = [
        "admiring", "adoring", "appreciative", "amused", "angry", "annoyed", "anxious", "awestruck", "uncomfortable",
        "bored", "calm", "focused", "contemplative", "confused", "contemptuous", "content", "hungry", "desirous",
        "determined", "disappointed", "disapproving", "disgusted", "distressed", "doubtful", "euphoric", "embarrassed",
        "disturbed", "enthusiastic", "entranced", "envious", "excited", "fearful", "grateful", "guilty", "horrified",
        "interested", "happy", "enamored", "nostalgic", "pained", "proud", "inspired", "relieved", "smitten", "sad",
        "satisfied", "desirous", "ashamed", "negatively surprised", "positively surprised", "sympathetic", "tired",
        "triumphant"
    ]

    @classmethod
    def scores_to_text(cls, emotion_scores: List[float]) -> str:
        if len(emotion_scores) == 48:
            adjectives = cls.ADJECTIVES_48
        elif len(emotion_scores) == 53:
            adjectives = cls.ADJECTIVES_53
        else:
            raise ValueError(f"Invalid length for emotion_scores {len(emotion_scores)}")

        # Return "neutral" if no emotions rate highly
        if all(emotion_score < cls.RANGES[0][0] for emotion_score in emotion_scores):
            return "neutral"

        # Construct phrases for all emotions that rate highly enough
        phrases = [""] * len(emotion_scores)
        for range_idx, (range_min, range_max) in enumerate(cls.RANGES):
            for emotion_idx, emotion_score in enumerate(emotion_scores):
                if range_min < emotion_score < range_max:
                    phrases[emotion_idx] = f"{cls.ADVERBS[range_idx]} {adjectives[emotion_idx]}"

        # Sort phrases by score
        sorted_indices = np.argsort(emotion_scores)[::-1]
        phrases = [phrases[i] for i in sorted_indices if phrases[i] != ""]

        # If there is only one phrase that rates highly, return it
        if len(phrases) == 0:
            return phrases[0]

        # Return all phrases separated by conjunctions
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1]