import gradio as gr
from chatbot_transformer import reply


# -- Gradio Interface (رابط کاربری گریدیو)
# تابع برای پاسخ به سوالات
# This function is used to answer questions.
# It takes user input, processes it, and returns a response.
def chat_fn(user_input):
    return reply(user_input)


# راه‌اندازی رابط کاربری گریدیو
# این بخش شامل راه‌اندازی رابط کاربری گریدیو است که به کاربران امکان می‌دهد سوالات خود را وارد کنند و پاسخ‌ها را دریافت کنند.
# This part includes setting up the Gradio interface that allows users to enter their questions and receive
gr.Interface(
    fn=chat_fn,
    inputs="text",
    outputs="text",
    title="Transformer Chatbot",
    description="پاسخ به سوالات با دیتاست فیلم‌ها"
).launch()
