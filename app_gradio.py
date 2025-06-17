import gradio as gr
from chatbot_transformer import reply

# -- Gradio Interface (رابط کاربری گریدیو)
# تابع برای پاسخ به سوالات  
def chat_fn(user_input):
    return reply(user_input)


# راه‌اندازی رابط کاربری گریدیو
gr.Interface(fn=chat_fn,
             inputs="text",
             outputs="text",
             title="Transformer Chatbot",
             description="پاسخ به سوالات با دیتاست فیلم‌ها").launch()
