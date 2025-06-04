# All sources used are given below.
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph

MONGODB_URI="mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "1"}} 


def main():
    
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        
        r = sr.Recognizer()
    
        with sr.Microphone() as source:
            while True:
                print("Say something!")
                r.adjust_for_ambient_noise(source) # listen for 1 second to calibrate the energy threshold for ambient noise levels
                r.pause_threshold = 3 # If we don't speak for 3 seconds, it's going to stop.

                audio = r.listen(source)
                
                print("Extracting text...")
                speech_to_text = r.recognize_google(audio)

                print("You said:", speech_to_text)
                
                for event in graph_with_mongo.stream({"messages": [{"role": "user", "content": speech_to_text}]}, config, stream_mode='values'): # Whenever we are using checkpointing, we need to pass 'config'
                    if "messages" in event:
                        event["messages"][-1].pretty_print()

main()

# Sources
# https://github.com/Uberi/speech_recognition/blob/master/examples/calibrate_energy_threshold.py