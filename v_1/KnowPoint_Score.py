import json
import openai


class DifficultyEvaluator:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        openai.api_base = 'https://api.gptapi.us/v1/chat/completions'

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取JSON字符串"""
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            return text
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            return text[start:end + 1]
        raise ValueError(f"No valid JSON found in response, 对应text: {text}")

    def evaluate_difficulty(self, keyword: str, knowledge_points: list) -> list:
        """
        调用 LLM API，为每个知识点评估最低适用年龄和理由。
        :param keyword: 当前关键词
        :param knowledge_points: 知识点列表
        :return: 评估结果列表
        """
        results = []
        prompt = f"""
        你是一名教育专家，任务是根据知识点的内容判断其最低适用年龄。以下是学生的认知能力划分模型：

        - 3-6岁：基础认知阶段（能理解简单的概念和具象内容）。
        - 7-9岁：初级逻辑推理阶段（能进行简单的分类、排序和因果推理）。
        - 10-12岁：中级逻辑推理阶段（能理解较复杂的关系和多步骤推理）。
        - 13-15岁：抽象思维和问题解决能力发展阶段（能理解抽象概念和复杂问题）。
        - 16-18岁：高级元认知阶段（具备自我反思、系统思考能力）。

        请对以下知识点进行分析(关键词是{keyword})：
        {json.dumps(knowledge_points[:3], ensure_ascii=False, indent=4)}

        输出以下 JSON 格式：
        [
            {{
                "point": "知识点内容",
                "最低适用年龄": 年龄,
                "reason": {{
                    "涉及内容": "本知识点涉及的内容，需要学生具备的能力。",
                    "适用理由": "为什么选择该年龄作为最低适用年龄。"
                }}
            }},
            ...
        ]
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an educational expert specializing in curriculum design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            content = self._extract_json(response['choices'][0]['message']['content'])
            print(content)
            results = json.loads(content)
        except Exception as e:
            print(f"Error evaluating difficulty for keyword '{keyword}': {e}")
        return results


def main():
    evaluator = DifficultyEvaluator(api_key="sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5")

    # 加载现有知识点
    with open('../results/knowledge_points.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for entry in data:
        keyword = entry["keyword"]
        knowledge_points = entry["knowledge_points"]
        print(f"正在评估关键词：{keyword}")

        # 按每次 3 个知识点分批处理
        batch_size = 3
        evaluated_points = []

        from tqdm import tqdm

        # 添加 tqdm 进度条
        with tqdm(total=len(knowledge_points), desc=f"处理关键词: {keyword}", unit="点") as pbar:
            for i in range(0, len(knowledge_points), batch_size):
                batch = knowledge_points[i:i + batch_size]  # 每次读取 3 个
                print(f"正在处理知识点批次：{batch}")
                batch_results = evaluator.evaluate_difficulty(keyword, batch)
                evaluated_points.extend(batch_results)

                # 更新进度条
                pbar.update(len(batch))

        results.append({"keyword": keyword, "evaluated_points": evaluated_points})

    # 保存新结果
    with open('../results/evaluated_knowledge_points.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        print("评估结果已保存至 evaluated_knowledge_points.json")


if __name__ == "__main__":
    main()

    def age_to_difficulty(age: int):
        if age <= 6:
            return 1
        elif age <= 9:
            return 3
        elif age <= 12:
            return 5
        elif age <= 15:
            return 7
        elif age <= 18:
            return 9
        else:
            raise ValueError(f'错误的年龄输入: {age}')


    with open('../results/evaluated_knowledge_points.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for d in data:
        evaluated_points = d['evaluated_points']
        for ep in evaluated_points:
            age = ep['最低适用年龄']
            cog = age_to_difficulty(age)
            ep['Difficulty'] = cog

    with open('../results/evaluated_knowledge_points.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print('Difficulty增加成功')

