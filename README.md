在本次工作中，算家计算以 DeepSeek-R1-Distill-Qwen-32B 预训练模型为基础，针对贵州地域特色和行业场景，构建了贵阳贵安语料数据集，并采用 LoRA技术进行了微调。

1.数据集构建

贵阳贵安数据集内容主要聚焦在6个领域：城市基础知识库、政务服务与政策法规、经济运行与产业特色、城市运行与服务、文化旅游、社会民生。

2.LoRA 微调设置

Adapter 结构：在 Q-Attention、Q-FFN 等关键层插入低秩矩阵（rank=8）作为可训练的微调参数，其余参数保持冻结；

学习率与优化器：采用 AdamW 优化器，初始学习率设置为 5e-4，权重衰减系数 0.01。




一、模型下载地址 

huggingface：
 https://huggingface.co/suanjiayun/DeepSeek-R1-Distill-Qwen-32B-lora/

modelscope： 
https://www.modelscope.cn/models/suanjia/DeepSeek-R1-Distill-Qwen-32B/

二、页面

 项目提供了一个简单的页面，运行webui.py即可 streamlit run webui.py --server.address 0.0.0.0 --server.port 8080

三、模型快速体验

可前往算家云快速体验：https://suanjiayun.com/
