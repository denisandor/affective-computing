# Emotion-Aware Music Player

This project is a small affective computing system that tries to look at how you feel and then choose music that fits or gently changes your mood.

The app combines two types of signals:
- Facial expressions from a webcam
- Vocal cues from short speech recordings

From these, the system predicts basic emotions such as happiness, sadness, anger, and neutrality.  
The predicted emotion is then mapped to a matching playlist through the Spotify API. Depending on the design, the music can either mirror the current mood (sad music when you are sad) or try to regulate it (calmer tracks when you are stressed, more energetic tracks when you are low).

## Project goals

- Detect a user's emotional state in close to real time using face and voice
- Use those emotions to drive adaptive music selection instead of static playlists
- Explore whether this kind of context aware playback feels more personal and helpful for focus, relaxation, or general wellbeing

## What the system does at a high level

1. Capture input  
   - Webcam frames for facial expressions  
   - Short audio clips for voice

2. Recognize emotion  
   - Run both signals through trained models to estimate the current emotion  
   - Combine the two predictions into a single final emotion label

3. Adapt music  
   - Map the final emotion to a specific playlist or track set  
   - Control Spotify playback accordingly
