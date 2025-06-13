from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import re

st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.chat-container {
  max-width: 800px;
  margin: auto;
  padding: 1rem;
}
.user-bubble, .assistant-bubble {
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  line-height: 1.5;
  width: fit-content;
  max-width: 80%;
}
.user-bubble {
  background: #e0e0e0;
  margin-left: auto;
}
.assistant-bubble {
  background: #f0f8ff;
  margin-right: auto;
}
.think-block {
  font-size: 0.9rem;
  color: #555;
  margin-top: 0.25rem;
  background: #fafafa;
  border-left: 3px solid #ccc;
  padding-left: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True
)

# 页面标题
st.markdown("# 💬 DeepSeek R1 Distill Chatbot")
st.markdown("🚀 Powered by 算家计算")

# 模型参数
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# 拆分“思考”与“回答”
def split_text(text):
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()

# 加载模型
@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tok, mdl

tokenizer, model = load_model()

# 会话历史储存
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "您好，请问有什么可以帮您？"}
    ]

# 聊天输入框（放在最前面以便采集新输入）
prompt = st.text_input("输入你的问题，按回车发送：", key="input")


if prompt:
    # 加入用户
    st.session_state.history.append({"role": "user", "content": prompt})

    # 构造模型输入
    input_text = tokenizer.apply_chat_template(
        st.session_state.history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # 模型生成（不限制长度）
    max_len = inputs.input_ids.shape[1] + model.config.max_position_embeddings
    out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_len,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    resp = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # 加入助手
    st.session_state.history.append({"role": "assistant", "content": resp})

# 渲染对话区域
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.history:
    role = msg["role"]
    text = msg["content"].replace("\n", "<br/>")
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{text}</div>", unsafe_allow_html=True)
    else:
        think, ans = split_text(msg["content"])
        if think:
            # 有思考过程
            st.markdown(f"<div class='assistant-bubble'>{ans.replace('\n','<br/>')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='think-block'>🧠 {think.replace('\n','<br/>')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{text}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
