在本次工作中，算家计算以 DeepSeek-R1-Distill-Qwen-32B 预训练模型为基础，针对贵州地域特色和行业场景，构建了贵阳贵安语料数据集，并采用 LoRA技术进行了微调。

1.数据集构建

贵阳贵安数据集内容主要聚焦在6个领域：城市基础知识库、政务服务与政策法规、经济运行与产业特色、城市运行与服务、文化旅游、社会民生。

2.LoRA 微调设置

Adapter 结构：在 Q-Attention、Q-FFN 等关键层插入低秩矩阵（rank=8）作为可训练的微调参数，其余参数保持冻结；

学习率与优化器：采用 AdamW 优化器，初始学习率设置为 5e-4，权重衰减系数 0.01。




一、模型下载地址 

huggingface：
https://huggingface.co/suanjia/DeepSeek-R1-Distill-Qwen-32B-GuiyangGuian-lora-v1/

modelscope： 
https://www.modelscope.cn/models/suanjia/DeepSeek-R1-Distill-Qwen-32B/

二、模型使用

1.使用
DeepSeek-R1-Distill 模型可以像使用 Qwen 或 Llama 模型一样使用。

例如，您可以使用 vLLM 轻松启动服务：

VLLM_USE_MODELSCOPE=true vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
您也可以使用 SGLang 轻松启动服务：

SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2

参考官方文档 https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B


2.页面

 项目提供了一个简单的页面，运行webui.py即可 streamlit run webui.py --server.address 0.0.0.0 --server.port 8080

三、模型快速体验

可前往算家云快速体验：https://suanjiayun.com/
