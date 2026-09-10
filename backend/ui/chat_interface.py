import gradio as gr
from main import init_chat


def greet(name, intensity):
    return "Hello " + name + "!" * int(intensity)

gr.ChatInterface(
    fn = init_chat,
    api_name="predict"
).launch()