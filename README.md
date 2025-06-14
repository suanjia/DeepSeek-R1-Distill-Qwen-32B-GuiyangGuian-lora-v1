在本次工作中，我们以 DeepSeek-R1-Distill-Qwen-32B 预训练模型为基础，针对贵州地域特色和行业场景，使用专门构建的贵州领域语料数据集，采用 LoRA（Low‐Rank Adaptation）技术进行了微调，具体过程包括：

1.数据集构建

收集来源：汇集了贵州本地新闻、政府公告、旅游攻略、电商评论和社交媒体对话等多种文本资源；

预处理：统一进行了分句、去重、清洗（剔除低质量、重复或含敏感信息的文本）以及分词标注。

2.LoRA 微调设置

Adapter 结构：在 Q-Attention、Q-FFN 等关键层插入低秩矩阵（rank=8）作为可训练的微调参数，其余参数保持冻结；

学习率与优化器：采用 AdamW 优化器，初始学习率设置为 5e-4，权重衰减系数 0.01；

3.微调效果评估

语言理解：在贵州地方制度等任务上，模型回答更具体 ；

文本生成：在地名和少数民族语言片段的连贯性测试中，相较于原模型生成错误率下降了；


一、模型下载地址 

huggingface：
 https://huggingface.co/suanjiayun/DeepSeek-R1-Distill-Qwen-32B-lora/

modelscope： 
https://www.modelscope.cn/models/suanjia/DeepSeek-R1-Distill-Qwen-32B/

二、页面

 项目提供了一个简单的页面，运行webui.py即可 streamlit run webui.py --server.address 0.0.0.0 --server.port 8080
