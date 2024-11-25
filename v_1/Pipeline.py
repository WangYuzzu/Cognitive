import json
from calculate_cog import StudentInfoExtractor, CognitionCalculator
from Prompt import PromptGenerator
import openai
from typing import Optional


class KnowledgeFlow:
    """
    知识点流转组件：串联自然语言描述到 Prompt 生成的全流程
    """

    def __init__(self, api_key: str, evaluated_knowledge_file: str):
        self.api_key = api_key
        self.knowledge_data = self._load_knowledge_data(evaluated_knowledge_file)
        self.extractor = StudentInfoExtractor(api_key)
        self.calculator = CognitionCalculator()
        self.prompt_generator = PromptGenerator()

    @staticmethod
    def _load_knowledge_data(file_path: str) -> dict:
        """
        加载 evaluated_knowledge_points.json 文件
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"知识点文件未找到：{file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"知识点文件格式错误：{file_path}")

    def get_knowledge_difficulty(self, keyword: str, knowledge_point: str) -> Optional[int]:
        """
        根据关键词和知识点内容检索难度值
        """
        for entry in self.knowledge_data:
            if entry["keyword"] == keyword:
                for point in entry["evaluated_points"]:
                    if point["point"] == knowledge_point:
                        return point.get("Difficulty")  # 假定 "Difficulty" 键存储难度值
        raise ValueError(f"未找到对应的知识点难度值，关键词：{keyword}, 知识点：{knowledge_point}")

    def process(self, description: str, keyword: str, knowledge_point: str) -> str:
        """
        流程：
        1. 提取学生认知信息
        2. 计算学生认知值
        3. 获取知识点难度值
        4. 生成 Prompt
        5. 调用 LLM API 获取结果
        """
        # Step 1: 提取学生信息
        student_info = self.extractor.call_llm_api(description)
        if not student_info:
            raise ValueError("未能成功提取学生信息，请检查输入描述")

        # Step 2: 计算学生认知值
        student_cognition = self.calculator.calculate_cognition(
            age=student_info["age"],
            region=student_info["region"],
            is_key_school=student_info["is_key_school"]
        )

        # Step 3: 获取知识点难度值
        knowledge_difficulty = self.get_knowledge_difficulty(keyword, knowledge_point)
        print(f'{student_cognition}-->{knowledge_difficulty}')
        # Step 4: 生成 Prompt
        prompt = self.prompt_generator.generate_prompt(student_cognition, knowledge_difficulty, knowledge_point)
        print(prompt)
        # Step 5: 调用 LLM API
        return self.call_llm_api(prompt)

    def call_llm_api(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用 LLM API 生成结果
        """
        openai.api_base = "https://api.gptapi.us/v1/chat/completions"
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a teacher who is good at determining how to explain the knowledge points "
                                                      "according to the difficulty of the knowledge points and the cognitive level of the students."},
                        {"role": "user", "content": prompt + '请记住，你面对的是学生，所以请直接按要求讲解'}
                    ],
                    temperature=0
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise e


# 测试主流程
def main():
    # 配置
    api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
    evaluated_knowledge_file = "../results/evaluated_knowledge_points.json"

    # 初始化
    knowledge_flow = KnowledgeFlow(api_key, evaluated_knowledge_file)

    # 输入数据
    description_0 = "我今年8岁，在漯河上学，尽管我希望在重点小学，但事与愿违"
    description_1 = "我今年8岁，在北京重点小学哦"
    description_2 = "我今年根号100岁，在漯河上学，尽管我希望在重点小学，但事与愿违"
    description_3 = "我今年14岁，在北京的重点中学读书"

    description_list = [description_0, description_1, description_2, description_3]
    keyword = "机器学习"
    knowledge_point = "什么是机器学习？"

    # 流程运行
    try:
        for description in description_list:
            result = knowledge_flow.process(description, keyword, knowledge_point)
            print("生成的结果：\n", result)
    except Exception as e:
        print(f"流程失败：{e}")


if __name__ == "__main__":
    main()
