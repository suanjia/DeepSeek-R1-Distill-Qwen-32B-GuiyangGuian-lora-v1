算家计算基于 DeepSeek-R1-Distill-Qwen-32B 预训练模型，深度结合贵州地域特色与行业应用场景，通过构建专业语料数据集，并运用 LoRA 技术进行微调，打造出适用于贵阳贵安的通用基础大模型。 

一、贵阳贵安语料数据集构
围绕贵州区域发展核心需求，贵阳贵安数据集涵盖六大核心领域： 

1.城市基础知识库：整合贵阳贵安地理、人口、规划等基础数据 

2.政务服务与政策法规：收录区域政策文件、办事指南等政务信息 

3.经济运行与产业特色：聚焦当地经济数据、特色产业发展动态 

4.城市运行与服务：覆盖交通、能源、公共服务等城市治理内容 

5.文化旅游：包含地方文化遗产、旅游资源、民俗风情等信息 

6.社会民生：涉及教育、医疗、就业等民生领域内容 

二、LoRA 微调策略 

采用低秩自适应（LoRA）技术进行模型优化，具体参数设置如下： 

1.Adapter 结构设计：在 Q-Attention、Q-FFN 等核心模块插入秩为 8 的低秩矩阵，将其作为可训练参数进行优化，同时冻结模型其他参数，在保证性能的同时大幅降低计算成本 

2.训练参数配置：选用 AdamW 优化器，设置初始学习率为 5e-4，权重衰减系数为 0.01，确保模型在微调过程中实现高效收敛

三、模型下载地址 

huggingface：

https://huggingface.co/suanjia/DeepSeek-R1-Distill-Qwen-32B-GuiyangGuian-lora-v1/

modelscope： 

https://www.modelscope.cn/models/suanjia/DeepSeek-R1-Distill-Qwen-32B/

四、模型使用

1.使用

DeepSeek-R1-Distill 模型可以像使用 Qwen 或 Llama 模型一样使用。

例如，您可以使用 vLLM 轻松启动服务：

```VLLM_USE_MODELSCOPE=true vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager```

您也可以使用 SGLang 轻松启动服务：

```SGLANG_USE_MODELSCOPE=true python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2```

参考官方文档 https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B


2.页面

 项目提供了一个简单的页面，运行webui.py即可 streamlit run webui.py --server.address 0.0.0.0 --server.port 8080

五、模型快速体验

可前往算家云快速体验：https://suanjiayun.com/

