import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 載入 GPT2 模型和 Tokenizer
model_name = "gpt2"  # 你可以選擇更強大的模型，如 "gpt-neo", "gpt-3" 等
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 定義生成運勢的函數
def generate_horoscope(zodiac_sign, style="幽默"):
    prompt = f"請為星座 {zodiac_sign} 生成今天的運勢，風格是 {style}，內容包括愛情、事業、財運和健康。"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    horoscope = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return horoscope

# 創建 Gradio 介面
iface = gr.Interface(
    fn=generate_horoscope,
    inputs=[
        gr.Textbox(placeholder="例如：牡羊座", label="輸入星座"),
        gr.Radio(
            ["正經", "幽默", "深刻", "神秘", "詩意", "激勵", "科幻", "搞笑", "懸疑", "哲學"], 
            label="選擇運勢風格"
        )
    ],
    outputs="text",
    title="每日星座運勢生成器",
    description="輸入你的星座，選擇運勢風格，來查看今天的運勢！"
)

# 啟動介面
iface.launch()
