import json
import openai
from tqdm import tqdm

# 设置 OpenAI API Key
openai.api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
openai.api_base = 'https://api.gptapi.us/v1/chat/completions'

def _extract_json(text: str) -> str:
    """从文本中提取JSON字符串"""
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        return text[start:end + 1]
    raise ValueError(f"No valid JSON found in response, 对应text: {text}")


def generate_knowledge_points(keyword):
    """调用 LLM API 为关键词生成知识点"""
    prompt = f"""
    你是一名教育领域的知识专家，任务是根据关键词生成相关的知识点列表。以下是具体要求：

    1. 根据给定的关键词生成与其相关的知识点，可以涵盖以下几个方面：
       - 基础概念
       - 核心原理
       - 实际应用
       - 延伸内容
    请注意：不仅限于以上4种，如果有更好的方面，你需要给出更好的。

    2. 输出以严格的 JSON 格式表示，格式示例如下：
    {{
      "keyword": "机器学习",
      "knowledge_points": [
        "什么是机器学习？",
        "机器学习的主要类型：监督学习、无监督学习、强化学习。",
        "常见的机器学习算法及其特点（如线性回归、决策树、支持向量机）。",
        "机器学习在图像识别中的应用。",
        "深度学习与传统机器学习的关系。",
        "机器学习的局限性：过拟合、数据依赖性。"
      ]
    }}

    请为以下关键词生成知识点：{keyword}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in generating structured educational content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = _extract_json(response['choices'][0]['message']['content'])
        print(content)
        return json.loads(content)  # 解析为 JSON 对象
    except Exception as e:
        print(f"Error generating knowledge points for keyword '{keyword}': {e}")
        return {"keyword": keyword, "knowledge_points": []}  # 返回空知识点列表以确保完整性


def main():
    # 读取关键词文件
    with open('../results/kg_state_graph.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        nodes_list = data['nodes']
        print(f'一共 {len(nodes_list)} 个关键词')

    # 创建新的数据结构用于存储知识点
    results = []

    # 遍历每个关键词生成知识点
    for keyword in tqdm(nodes_list[:10], desc='进度'):
        if keyword:
            print(f"正在生成关键词: {keyword}")
            knowledge_data = generate_knowledge_points(keyword)
            results.append(knowledge_data)

    # 保存到新的 JSON 文件
    with open('../results/knowledge_points.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        print("知识点已保存到 ../results/knowledge_points.json")


if __name__ == "__main__":
    main()
